// -*- C++ -*-
//
// Package:    PFTracking
// Class:      GoodSeedProducer
// 
// Original Author:  Michele Pioppi

#include "RecoParticleFlow/PFTracking/interface/GoodSeedProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"  
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"

#include <fstream>
#include <string>

using namespace edm;
using namespace std;
PFResolutionMap* GoodSeedProducer::resMapEtaECAL_ = 0;                                        
PFResolutionMap* GoodSeedProducer::resMapPhiECAL_ = 0;

GoodSeedProducer::GoodSeedProducer(const ParameterSet& iConfig):
  maxShPropagator_(0),
  pfTransformer_(0),
  conf_(iConfig)
{
  LogInfo("GoodSeedProducer")<<"Electron PreIdentification started  ";

  //now do what ever initialization is needed
 
  tracksContainers_ = 
    iConfig.getParameter< vector < InputTag > >("TkColList");

  pfCLusTagECLabel_=
    iConfig.getParameter<InputTag>("PFEcalClusterLabel");

  pfCLusTagPSLabel_=
    iConfig.getParameter<InputTag>("PFPSClusterLabel");

  preidgsf_=iConfig.getParameter<string>("PreGsfLabel");
  preidckf_=iConfig.getParameter<string>("PreCkfLabel");

  propagatorName_ = iConfig.getParameter<string>("Propagator");
  fitterName_ = iConfig.getParameter<string>("Fitter");
  smootherName_ = iConfig.getParameter<string>("Smoother");
  


  nHitsInSeed_=iConfig.getParameter<int>("NHitsInSeed");

  clusThreshold_=iConfig.getParameter<double>("ClusterThreshold");

  //collection to produce
  produceCkfseed_ = iConfig.getUntrackedParameter<bool>("ProduceCkfSeed",false);
  produceCkfPFT_ = iConfig.getUntrackedParameter<bool>("ProduceCkfPFTracks",true);  

  LogDebug("GoodSeedProducer")<<"Seeds for GSF will be produced ";
  produces<TrajectorySeedCollection>(preidgsf_);

  if(produceCkfseed_){
    LogDebug("GoodSeedProducer")<<"Seeds for CKF will be produced ";
   produces<TrajectorySeedCollection>(preidckf_);
  }

  if(produceCkfPFT_){
    LogDebug("GoodSeedProducer")<<"PFTracks from CKF tracks will be produced ";
    produces<reco::PFRecTrackCollection>();
  }


  useTmva_= iConfig.getUntrackedParameter<bool>("UseTMVA",false);
}


GoodSeedProducer::~GoodSeedProducer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.) 
  delete maxShPropagator_;
  delete pfTransformer_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GoodSeedProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
 
  LogDebug("GoodSeedProducer")<<"START event: "<<iEvent.id().event()
			      <<" in run "<<iEvent.id().run();

  auto_ptr<TrajectorySeedCollection> output_preid(new TrajectorySeedCollection);
  auto_ptr<TrajectorySeedCollection> output_nopre(new TrajectorySeedCollection);
  auto_ptr< reco::PFRecTrackCollection > 
    pOutputPFRecTrackCollection(new reco::PFRecTrackCollection);







			      
  Handle<reco::PFClusterCollection> theECPfClustCollection;
  iEvent.getByLabel(pfCLusTagECLabel_,theECPfClustCollection);
  Handle<reco::PFClusterCollection> thePSPfClustCollection;
  iEvent.getByLabel(pfCLusTagPSLabel_,thePSPfClustCollection);
  vector<reco::PFCluster> basClus;
  vector<reco::PFCluster>::const_iterator iklus;
  for (iklus=theECPfClustCollection.product()->begin();
       iklus!=theECPfClustCollection.product()->end();
       iklus++){
    if((*iklus).energy()>clusThreshold_) basClus.push_back(*iklus);
  }
  
  ps1Clus.clear();
  ps2Clus.clear();

  for (iklus=thePSPfClustCollection.product()->begin();
       iklus!=thePSPfClustCollection.product()->end();
       iklus++){
    //layer==-11 first layer of PS
    //layer==-12 secon layer of PS
    if ((*iklus).layer()==-11) ps1Clus.push_back(*iklus);
    if ((*iklus).layer()==-12) ps2Clus.push_back(*iklus);
  }


  for (uint istr=0; istr<tracksContainers_.size();istr++){
    
    Handle<reco::TrackCollection> tkRefCollection;
    iEvent.getByLabel(tracksContainers_[istr], tkRefCollection);
    reco::TrackCollection  Tk=*(tkRefCollection.product());
    
    Handle<vector<Trajectory> > tjCollection;
    iEvent.getByLabel(tracksContainers_[istr], tjCollection);
    vector<Trajectory> Tj=*(tjCollection.product());

   
    LogDebug("GoodSeedProducer")<<"Number of tracks in colloction "
                                <<tracksContainers_[istr] <<" to be analyzed "
                                <<Tj.size();


 

  for(uint i=0;i<Tk.size();i++){
    int ipteta=getBin(Tk[i].eta(),Tk[i].pt());

    int ibin=ipteta*9;
    reco::TrackRef trackRef(tkRefCollection, i);
    TrajectorySeed Seed=Tj[i].seed();

    float PTOB=Tj[i].lastMeasurement().updatedState().globalMomentum().mag();
    float chikfred=Tk[i].normalizedChi2();
    int nhitpi=Tj[i].foundHits();
    float EP=0;


    //CLUSTERS - TRACK matching
      
    int side=0;  
    TSOS ecalTsos=
      pfTransformer_->getStateOnSurface(PFGeometry::ECALInnerWall,
					Tj[i].firstMeasurement().updatedState(),
					propagator_.product(),side);  
    
    float toteta=1000;
    float totphi=1000;
    float dr=1000;
    float EE=0;
    float feta=0;

    if(ecalTsos.isValid()){
      float etarec=ecalTsos.globalPosition().eta();
      float phirec=ecalTsos.globalPosition().phi();
      for(vector<reco::PFCluster>::const_iterator aClus = basClus.begin();
	  aClus != basClus.end(); aClus++) {
	
	
	ReferenceCountingPointer<Surface> showerMaxWall=
	  pfTransformer_->showerMaxSurface(aClus->energy(),true,ecalTsos,side);
	
	if (&(*showerMaxWall)!=0){
	  TSOS maxShTsos= maxShPropagator_->propagate
	    (*(ecalTsos.freeTrajectoryState()), *showerMaxWall);
	  
	  
	  if (maxShTsos.isValid()){
	    etarec=maxShTsos.globalPosition().eta();
	    phirec=maxShTsos.globalPosition().phi();
	  }
	}
	
	float tmp_dr=sqrt(pow((aClus->positionXYZ().phi()-phirec),2)+
			  pow((aClus->positionXYZ().eta()-etarec),2));
	if (tmp_dr<dr) {
	  dr=tmp_dr;
	  toteta=aClus->positionXYZ().eta()-etarec;
	  totphi=aClus->positionXYZ().phi()-phirec;
	  EP=aClus->energy()/PTOB;
	  EE=aClus->energy();
	  feta= aClus->positionXYZ().eta();
	}
      }
    }

    double ecaletares 
      = resMapEtaECAL_->GetBinContent(resMapEtaECAL_->FindBin(feta,EE));
    double ecalphires 
      = resMapPhiECAL_->GetBinContent(resMapPhiECAL_->FindBin(feta,EE)); 

    float chieta=(toteta!=1000)? toteta/ecaletares : toteta;
    float chiphi=(totphi!=1000)? totphi/ecalphires : totphi;
    float chichi= sqrt(chieta*chieta + chiphi*chiphi);

    bool cc1= false;

    //TMVA Analysis
    if(useTmva_){
      chi=chichi;
      chired=1000;
      chiRatio=1000;
      dpt=0;
      nhit=nhitpi;

      Trajectory::ConstRecHitContainer tmp;
      Trajectory::ConstRecHitContainer hits=Tj[i].recHits();
      for (int ih=hits.size()-1; ih>=0; ih--)  tmp.push_back(hits[ih]);
      vector<Trajectory> FitTjs=(fitter_.product())->fit(Seed,tmp,Tj[i].lastMeasurement().updatedState());
	
      if(FitTjs.size()>0){
	if(FitTjs[0].isValid()){
	  vector<Trajectory> SmooTjs=(smoother_.product())->trajectories(FitTjs[0]);
	  if(SmooTjs.size()>0){
	    if(SmooTjs[0].isValid()){
		
	      //Track refitted with electron hypothesis
		
	      float pt_out=SmooTjs[0].firstMeasurement().
		updatedState().globalMomentum().perp();
	      float pt_in=SmooTjs[0].lastMeasurement().
		updatedState().globalMomentum().perp();
	      dpt=(pt_in>0) ? fabs(pt_out-pt_in)/pt_in : 0.;
	      chiRatio=SmooTjs[0].chiSquared()/Tj[i].chiSquared();
	      chired=chiRatio*chikfred;
	    }
	  }
	}
      }
      eta=Tk[i].eta();
      pt=Tk[i].pt();
      eP=EP;
      //ENDCAP
      //USE OF PRESHOWER 
      if (fabs(Tk[i].eta())>1.6){
	ps2En=0;ps1En=0;
	ps2chi=100.; ps1chi=100.;
	PSforTMVA(Tj[i].firstMeasurement().updatedState());
      }

      
      float Ytmva=(fabs(Tk[i].eta())<1.6) ? 
	reader->EvaluateMVA( metBarrel_ ):
	reader->EvaluateMVA( metEndcap_ );
      
      if ( Ytmva>thrTMVA[ipteta]) cc1=true;
    }else{ 
      
      //thresholds     
      float chi2cut=thr[ibin+0];
      float ep_cutmin=thr[ibin+1];
      //
      int hit1max=int(thr[ibin+2]);
      float chiredmin=thr[ibin+3];
      //
      float chiratiocut=thr[ibin+4]; 
      float gschicut=thr[ibin+5]; 
      float gsptmin=thr[ibin+6];
      // 
      int hit2max=int(thr[ibin+7]);
      float finchicut=thr[ibin+8]; 
      //
      
      
      
      
      //Matching criteria
      bool aa1= (chichi<chi2cut);
      bool aa2= ((EP>ep_cutmin)&&(EP<1.2));
      bool aa3= (aa1 && aa2);
      int ipsbin=(ibin>=90)? 4*((ibin/9)-10) :-1;
      bool aa4= ((aa3)||(ibin<90))? false : PSCorrEnergy(Tj[i].firstMeasurement().updatedState(),ipsbin);
      bool aa5= (aa3 || aa4);
      //KF filter
      bool bb1 =
	((chikfred>chiredmin) || (nhitpi<hit1max));

      bool bb2 = false;
      bool bb3 = false;

      if((!aa5)&&bb1){

	Trajectory::ConstRecHitContainer tmp;
	Trajectory::ConstRecHitContainer hits=Tj[i].recHits();
	for (int ih=hits.size()-1; ih>=0; ih--)  tmp.push_back(hits[ih]);
	
	vector<Trajectory> FitTjs=(fitter_.product())->fit(Seed,tmp,Tj[i].lastMeasurement().updatedState());
	
	if(FitTjs.size()>0){
	  if(FitTjs[0].isValid()){
	    vector<Trajectory> SmooTjs=(smoother_.product())->trajectories(FitTjs[0]);
	    if(SmooTjs.size()>0){
	      if(SmooTjs[0].isValid()){
		
		//Track refitted with electron hypothesis
		
		float pt_out=SmooTjs[0].firstMeasurement().
		  updatedState().globalMomentum().perp();
		float pt_in=SmooTjs[0].lastMeasurement().
		  updatedState().globalMomentum().perp();
		float el_dpt=(pt_in>0) ? fabs(pt_out-pt_in)/pt_in : 0.;
		float chiratio=SmooTjs[0].chiSquared()/Tj[i].chiSquared();
		float gchi=chiratio*chikfred;
		
		//Criteria based on electron tracks
		bb2=((el_dpt>gsptmin)&&(gchi<gschicut)&&(chiratio<chiratiocut));
		bb3=((gchi<finchicut)&&(nhitpi<hit2max));
	      }
	    }
	  }
	}
      }
  

      
      bool bb4=(bb2 || bb3);
      bool bb5=(bb4 && bb1);
      cc1=(aa5 || bb5); 
      
      if(cc1)
	LogDebug("GoodSeedProducer")<<"Track (pt= "<<Tk[i].pt()<<
	  "GeV/c, eta= "<<Tk[i].eta() <<
	  ") preidentified for agreement between  track and ECAL cluster";
      if(cc1 &&(!bb1))
	LogDebug("GoodSeedProducer")<<"Track (pt= "<<Tk[i].pt()<<
	  "GeV/c, eta= "<<Tk[i].eta() <<
	  ") preidentified only for track properties";
      
    }
 
    if (cc1){
      
      //NEW SEED with n hits
      int nHitsinSeed= min(nHitsInSeed_,Tj[i].foundHits());
      
      Trajectory seedTraj;
      OwnVector<TrackingRecHit>  rhits;

      vector<TrajectoryMeasurement> tm=Tj[i].measurements();

      for (uint ihit=tm.size()-1; ihit>=tm.size()-nHitsinSeed;ihit-- ){ 
	//for the first n measurement put the TM in the trajectory
	// and save the corresponding hit
	if ((*tm[ihit].recHit()).hit()->clone()->isValid()){
	  seedTraj.push(tm[ihit]);
	  rhits.push_back((*tm[ihit].recHit()).hit()->clone());
	}
      }
      PTrajectoryStateOnDet* state = TrajectoryStateTransform().
	persistentState(seedTraj.lastMeasurement().updatedState(),
			(*seedTraj.lastMeasurement().recHit()).hit()->geographicalId().rawId());
      
      TrajectorySeed NewSeed(*state,rhits,alongMomentum);

      output_preid->push_back(NewSeed);
      delete state;
      if(produceCkfPFT_){
	
	reco::PFRecTrack pftrack( trackRef->charge(), 
				  reco::PFRecTrack::KF_ELCAND, 
				  i, trackRef );
	
	bool valid = pfTransformer_->addPoints( pftrack, *trackRef, Tj[i] );
	if(valid)
	  pOutputPFRecTrackCollection->push_back(pftrack);		
      } 
    }else{
      if (produceCkfseed_){
	output_nopre->push_back(Seed);
      }
      if(produceCkfPFT_){
		
	reco::PFRecTrack pftrack( trackRef->charge(), 
				  reco::PFRecTrack::KF, 
				  i, trackRef );

	bool valid = pfTransformer_->addPoints( pftrack, *trackRef, Tj[i] );
	
	if(valid)
	  pOutputPFRecTrackCollection->push_back(pftrack);		
	
      }
    }
  }
  }
  if(produceCkfPFT_){
    iEvent.put(pOutputPFRecTrackCollection);
  }
  
  iEvent.put(output_preid,preidgsf_);
  if (produceCkfseed_)
    iEvent.put(output_nopre,preidckf_);
  
}
// ------------ method called once each job just before starting event loop  ------------
void 
GoodSeedProducer::beginJob(const EventSetup& es)
{

  //Get Magnetic Field
  ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);
  pfTransformer_= new PFTrackTransformer(magField.product());

  //Tracking Tools
  es.get<TrackingComponentsRecord>().get(propagatorName_, propagator_);
  es.get<TrackingComponentsRecord>().get(fitterName_, fitter_);
  es.get<TrackingComponentsRecord>().get(smootherName_, smoother_);
  maxShPropagator_=new StraightLinePropagator(magField.product());

  
  //Resolution maps
  FileInPath ecalEtaMap(conf_.getParameter<string>("EtaMap"));
  FileInPath ecalPhiMap(conf_.getParameter<string>("PhiMap"));
  resMapEtaECAL_ = new PFResolutionMap("ECAL_eta",ecalEtaMap.fullPath().c_str());
  resMapPhiECAL_ = new PFResolutionMap("ECAL_phi",ecalPhiMap.fullPath().c_str());

  if(useTmva_){
    reader = new TMVA::Reader();
    string Method = conf_.getParameter<string>("TMVAMethod");
    
    reader->AddVariable("eta",&eta);
    reader->AddVariable("pt",&pt);
    reader->AddVariable("eP",&eP);
    reader->AddVariable("chi",&chi);
    reader->AddVariable("nhit",&nhit);
    reader->AddVariable("chired",&chired);
    reader->AddVariable("chiRatio",&chiRatio);
    reader->AddVariable("dpt",&dpt);
    reader->AddVariable("ps1En",&ps1En);
    reader->AddVariable("ps2En",&ps2En);
    reader->AddVariable("ps1chi",&ps1chi);
    reader->AddVariable("ps2chi",&ps2chi);
    
    FileInPath BarrelWeigths(conf_.getParameter<string>("WeightsForBarrel"));
    metBarrel_=Method + " barrel";
    reader->BookMVA( metBarrel_, BarrelWeigths.fullPath().c_str()  );
    FileInPath EndcapWeigths(conf_.getParameter<string>("WeightsForEndcap"));
    metEndcap_=Method + " endcap";
    reader->BookMVA( metEndcap_, EndcapWeigths.fullPath().c_str()  );	
    
    FileInPath parTMVAFile(conf_.getParameter<string>("TMVAThresholdFile"));
    ifstream ifsTMVA(parTMVAFile.fullPath().c_str());
    for (int iy=0;iy<15;iy++) ifsTMVA >> thrTMVA[iy];
  }else{

    
    //read threshold
    FileInPath parFile(conf_.getParameter<string>("ThresholdFile"));
    ifstream ifs(parFile.fullPath().c_str());
    for (int iy=0;iy<135;iy++) ifs >> thr[iy];
    
    //read PS threshold
    FileInPath parPSFile(conf_.getParameter<string>("PSThresholdFile"));
    ifstream ifsPS(parPSFile.fullPath().c_str());
    for (int iy=0;iy<20;iy++) ifsPS >> thrPS[iy];
  }
}

int GoodSeedProducer::getBin(float eta, float pt){
  int ie=0;
  int ip=0;
  if (fabs(eta)<0.8) ie=0;
  else{ if (fabs(eta)<1.65) ie=1;
    else ie=2;
  }
  if (pt<2) ip=0;
  else {  if (pt<4) ip=1;
    else {  if (pt<6) ip=2;
      else {  if (pt<15) ip=3;
	else ip=4;
      }
    }
  }
  int iep= ie*5+ip;
  LogDebug("GoodSeedProducer")<<"Track pt ="<<pt<<" eta="<<eta<<" bin="<<iep;
  return iep;
}

bool GoodSeedProducer::PSCorrEnergy(const TSOS tsos, int ibin){
  int sder=0;
  float psEn1 =thrPS[ibin+0];
  float dX1   =thrPS[ibin+1];
  float psEn2 =thrPS[ibin+2];
  float dY2   =thrPS[ibin+3];
  TSOS ps1TSOS =
    pfTransformer_->getStateOnSurface(PFGeometry::PS1Wall, tsos,
                                      propagator_.product(), sder);
  if (!(ps1TSOS.isValid())) return false;
  GlobalPoint v1=ps1TSOS.globalPosition();
  
  if (!((v1.perp() >=
         PFGeometry::innerRadius(PFGeometry::PS1)) &&
        (v1.perp() <=
         PFGeometry::outerRadius(PFGeometry::PS1)))) return false;
  
  bool Ps1g=false;  
  vector<reco::PFCluster>::const_iterator ips;
  for (ips=ps1Clus.begin(); ips!=ps1Clus.end();ips++){
    if ((fabs(v1.x()-(*ips).positionXYZ().x())<dX1)&&
        (fabs(v1.y()-(*ips).positionXYZ().y())<7)&&
        (v1.z()*(*ips).positionXYZ().z()>0)&&
	((*ips).energy()>psEn1)) Ps1g=true;
  }
  if (!Ps1g) return false;
  TSOS ps2TSOS =
    pfTransformer_->getStateOnSurface(PFGeometry::PS2Wall, ps1TSOS,
                                      propagator_.product(), sder);
  if (!(ps2TSOS.isValid())) return false;
  GlobalPoint v2=ps2TSOS.globalPosition();
  
  if (!((v2.perp() >=
         PFGeometry::innerRadius(PFGeometry::PS2)) &&
        (v2.perp() <=
         PFGeometry::outerRadius(PFGeometry::PS2)))) return false;
  bool Ps2g =false;
  for (ips=ps2Clus.begin(); ips!=ps2Clus.end();ips++){
    if ((fabs(v2.x()-(*ips).positionXYZ().x())<10.)&&
	(fabs(v2.y()-(*ips).positionXYZ().y())<dY2)&&
	(v2.z()*(*ips).positionXYZ().z()>0)&&
	((*ips).energy()>psEn2)) Ps2g=true;    
  }

  return Ps2g;
  
}
void GoodSeedProducer::PSforTMVA(const TSOS tsos){
  int sder=0;
  TSOS ps1TSOS =
    pfTransformer_->getStateOnSurface(PFGeometry::PS1Wall, tsos,
                                      propagator_.product(), sder);
  if (ps1TSOS.isValid()){
    GlobalPoint v1=ps1TSOS.globalPosition();
  
    if ((v1.perp() >=
	 PFGeometry::innerRadius(PFGeometry::PS1)) &&
	(v1.perp() <=
	 PFGeometry::outerRadius(PFGeometry::PS1))) {
      float enPScl1=0;
      float chi1=100;
      vector<reco::PFCluster>::const_iterator ips;
      for (ips=ps1Clus.begin(); ips!=ps1Clus.end();ips++){
	float ax=((*ips).positionXYZ().x()-v1.x())/0.114;
	float ay=((*ips).positionXYZ().y()-v1.y())/2.43;
	float pschi= sqrt(ax*ax+ay*ay);
	if (pschi<chi1){
	  chi1=pschi;
	  enPScl1=(*ips).energy();
	}
      }
      ps1En=enPScl1;
      ps1chi=chi1;

      TSOS ps2TSOS =
	pfTransformer_->getStateOnSurface(PFGeometry::PS2Wall, ps1TSOS,
					  propagator_.product(), sder);
      if (ps2TSOS.isValid()){
	GlobalPoint v2=ps2TSOS.globalPosition();
	if ((v2.perp() >=
	     PFGeometry::innerRadius(PFGeometry::PS2)) &&
	    (v2.perp() <=
	     PFGeometry::outerRadius(PFGeometry::PS2))){
	  float enPScl2=0;
	  float chi2=100;
	  for (ips=ps2Clus.begin(); ips!=ps2Clus.end();ips++){
	    float ax=((*ips).positionXYZ().x()-v2.x())/1.88;
	    float ay=((*ips).positionXYZ().y()-v2.y())/0.1449;
	    float pschi= sqrt(ax*ax+ay*ay);
	    if (pschi<chi2){
	      chi2=pschi;
	      enPScl2=(*ips).energy();
	    }
	  }

	  ps2En=enPScl2;
	  ps2chi=chi2;

	}
      }
      
    }
    
  }
}
