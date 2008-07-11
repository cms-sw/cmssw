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
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include <fstream>
#include <string>
#include "TMath.h"

using namespace edm;
using namespace std;
using namespace reco;
PFResolutionMap* GoodSeedProducer::resMapEtaECAL_ = 0;                                        
PFResolutionMap* GoodSeedProducer::resMapPhiECAL_ = 0;

GoodSeedProducer::GoodSeedProducer(const ParameterSet& iConfig):
  pfTransformer_(0),
  conf_(iConfig)
{
  LogInfo("GoodSeedProducer")<<"Electron PreIdentification started  ";
  
  //now do what ever initialization is needed
 
  tracksContainers_ = 
    iConfig.getParameter< vector < InputTag > >("TkColList");
  
  minPt_=iConfig.getParameter<double>("MinPt");
  maxPt_=iConfig.getParameter<double>("MaxPt");
  maxEta_=iConfig.getParameter<double>("MaxEta");
  
  pfCLusTagECLabel_=
    iConfig.getParameter<InputTag>("PFEcalClusterLabel");
  
  pfCLusTagPSLabel_=
    iConfig.getParameter<InputTag>("PFPSClusterLabel");
  
  preidgsf_=iConfig.getParameter<string>("PreGsfLabel");
  preidckf_=iConfig.getParameter<string>("PreCkfLabel");
  
  
  fitterName_ = iConfig.getParameter<string>("Fitter");
  smootherName_ = iConfig.getParameter<string>("Smoother");
  
  
  
  nHitsInSeed_=iConfig.getParameter<int>("NHitsInSeed");

  clusThreshold_=iConfig.getParameter<double>("ClusterThreshold");
  
  minEp_=iConfig.getParameter<double>("MinEOverP");
  maxEp_=iConfig.getParameter<double>("MaxEOverP");

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
    produces<PFRecTrackCollection>();
  }


  useQuality_   = iConfig.getParameter<bool>("UseQuality");
  trackQuality_=TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));

  useTmva_= iConfig.getUntrackedParameter<bool>("UseTMVA",false);
}


GoodSeedProducer::~GoodSeedProducer()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.) 
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
  
  //Create empty output collections
  auto_ptr<TrajectorySeedCollection> output_preid(new TrajectorySeedCollection);
  auto_ptr<TrajectorySeedCollection> output_nopre(new TrajectorySeedCollection);
  auto_ptr< PFRecTrackCollection > 
    pOutputPFRecTrackCollection(new PFRecTrackCollection);
  
  
  //Tracking Tools
  iSetup.get<TrackingComponentsRecord>().get(fitterName_, fitter_);
  iSetup.get<TrackingComponentsRecord>().get(smootherName_, smoother_);

  //Handle input collections
  //ECAL clusters	      
  Handle<PFClusterCollection> theECPfClustCollection;
  iEvent.getByLabel(pfCLusTagECLabel_,theECPfClustCollection);
  
  vector<PFCluster> basClus;
  vector<PFCluster>::const_iterator iklus;
  for (iklus=theECPfClustCollection.product()->begin();
       iklus!=theECPfClustCollection.product()->end();
       iklus++){
    if((*iklus).energy()>clusThreshold_) basClus.push_back(*iklus);
  }
  
  //PS clusters
  Handle<PFClusterCollection> thePSPfClustCollection;
  iEvent.getByLabel(pfCLusTagPSLabel_,thePSPfClustCollection);
  
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
  
  //Vector of track collections
  for (uint istr=0; istr<tracksContainers_.size();istr++){
    
    //Track collection
    Handle<TrackCollection> tkRefCollection;
    iEvent.getByLabel(tracksContainers_[istr], tkRefCollection);
    TrackCollection  Tk=*(tkRefCollection.product());
    
    //Trajectory collection
    Handle<vector<Trajectory> > tjCollection;
    iEvent.getByLabel(tracksContainers_[istr], tjCollection);
    vector<Trajectory> Tj=*(tjCollection.product());
    
    
    LogDebug("GoodSeedProducer")<<"Number of tracks in collection "
                                <<tracksContainers_[istr] <<" to be analyzed "
                                <<Tj.size();
    

    //loop over the track collection
    for(uint i=0;i<Tk.size();i++){		
      if (useQuality_ &&
	  (!(Tk[i].quality(trackQuality_)))) continue;
      int ipteta=getBin(Tk[i].eta(),Tk[i].pt());
      int ibin=ipteta*8;
      TrackRef trackRef(tkRefCollection, i);
      TrajectorySeed Seed=Tj[i].seed();
      
      float PTOB=Tj[i].lastMeasurement().updatedState().globalMomentum().mag();
      float chikfred=Tk[i].normalizedChi2();
      int nhitpi=Tj[i].foundHits();
      float EP=0;
      
      
      //CLUSTERS - TRACK matching
      
      float pfmass=  0.0005;
      float pfoutenergy=sqrt((pfmass*pfmass)+Tk[i].outerMomentum().Mag2());
      XYZTLorentzVector mom =XYZTLorentzVector(Tk[i].outerMomentum().x(),
					       Tk[i].outerMomentum().y(),
					       Tk[i].outerMomentum().z(),
					       pfoutenergy);
      XYZTLorentzVector pos =   XYZTLorentzVector(Tk[i].outerPosition().x(),
						  Tk[i].outerPosition().y(),
						  Tk[i].outerPosition().z(),
						  0.);

      BaseParticlePropagator theOutParticle = 
	BaseParticlePropagator( RawParticle(mom,pos),
				0,0,B_.z());
      theOutParticle.setCharge(Tk[i].charge());
      
      theOutParticle.propagateToEcalEntrance(false);
      
      
      float toteta=1000;
      float totphi=1000;
      float dr=1000;
      float EE=0;
      float feta=0;
      
      if(theOutParticle.getSuccess()!=0){
	bool isBelowPS=(fabs(theOutParticle.vertex().eta())>1.65) ? true :false;	
	
	for(vector<PFCluster>::const_iterator aClus = basClus.begin();
	    aClus != basClus.end(); aClus++) {
	  double ecalShowerDepth
	    = PFCluster::getDepthCorrection(aClus->energy(),
						  isBelowPS,
						  false);
	  
	  math::XYZPoint meanShower=math::XYZPoint(theOutParticle.vertex())+
	    math::XYZTLorentzVector(theOutParticle.momentum()).Vect().Unit()*ecalShowerDepth;	
	  
	  float etarec=meanShower.eta();
	  float phirec=meanShower.phi();
	  float tmp_ep=aClus->energy()/PTOB;
          float tmp_phi=fabs(aClus->position().phi()-phirec);
	  if (tmp_phi>TMath::TwoPi()) tmp_phi-= TMath::TwoPi();
	  float tmp_dr=sqrt(pow(tmp_phi,2)+
			    pow((aClus->position().eta()-etarec),2));
	  
	  if ((tmp_dr<dr)&&(tmp_ep>minEp_)&&(tmp_ep<maxEp_)){
	    dr=tmp_dr;
	    toteta=aClus->position().eta()-etarec;
	    totphi=tmp_phi;
	    EP=tmp_ep;
	    EE=aClus->energy();
	    feta= aClus->position().eta();
	  }
	}
      }

      //Resolution maps
      double ecaletares 
	= resMapEtaECAL_->GetBinContent(resMapEtaECAL_->FindBin(feta,EE));
      double ecalphires 
	= resMapPhiECAL_->GetBinContent(resMapPhiECAL_->FindBin(feta,EE)); 
      
      //geomatrical compatibility
      float chieta=(toteta!=1000)? toteta/ecaletares : toteta;
      float chiphi=(totphi!=1000)? totphi/ecalphires : totphi;
      float chichi= sqrt(chieta*chieta + chiphi*chiphi);
      
      
      //Matching criteria
      float chi2cut=thr[ibin+0];
      float ep_cutmin=thr[ibin+1];
      bool GoodMatching= ((chichi<chi2cut) &&(EP>ep_cutmin) && (nhitpi>10));
      if (Tk[i].pt()>maxPt_) GoodMatching=true;
      if (Tk[i].pt()<minPt_) GoodMatching=false;
      //ENDCAP
      //USE OF PRESHOWER 
      if (fabs(Tk[i].eta())>1.68){
        int iptbin =4*getBin(Tk[i].pt());
	ps2En=0;ps1En=0;
	ps2chi=100.; ps1chi=100.;
	PSforTMVA(mom,pos);
	float p1e=thrPS[iptbin];
        float p2e=thrPS[iptbin+1];
        float p1c=thrPS[iptbin+2];
        float p2c=thrPS[iptbin+3];
	bool GoodPSMatching= 
	  ((ps2En>p2e)
	   &&(ps1En>p1e)
	   &&(ps1chi<p1c)
	   &&(ps2chi<p2c));
	GoodMatching = (GoodMatching && GoodPSMatching);
      }
      
      
      bool GoodRange= ((fabs(Tk[i].eta())<maxEta_) && 
                       (Tk[i].pt()>minPt_));
      //KF FILTERING FOR UNMATCHED EVENTS
      int hit1max=int(thr[ibin+2]);
      float chiredmin=thr[ibin+3];
      bool GoodKFFiltering =
	((chikfred>chiredmin) || (nhitpi<hit1max));
      
      bool GoodTkId= false;
      
      if((!GoodMatching) &&(GoodKFFiltering) &&(GoodRange)){

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
	
	//TMVA Analysis
	if(useTmva_){
	  
	  
	
	  eta=Tk[i].eta();
	  pt=Tk[i].pt();
	  eP=EP;

	  
	  
	  float Ytmva=reader->EvaluateMVA( method_ );
	  
	  float BDTcut=thr[ibin+4]; 
	  if ( Ytmva>BDTcut) GoodTkId=true;
	}else{ 
	  
	  
	  
	  //
	  float chiratiocut=thr[ibin+5]; 
	  float gschicut=thr[ibin+6]; 
	  float gsptmin=thr[ibin+7];

	  GoodTkId=((dpt>gsptmin)&&(chired<gschicut)&&(chiRatio<chiratiocut));      
       
	}
      }
    
      bool GoodPreId= (GoodTkId || GoodMatching); 
   

      
      if(GoodPreId)
	LogDebug("GoodSeedProducer")<<"Track (pt= "<<Tk[i].pt()<<
	  "GeV/c, eta= "<<Tk[i].eta() <<
	  ") preidentified for agreement between  track and ECAL cluster";
      if(GoodPreId &&(!GoodMatching))
	LogDebug("GoodSeedProducer")<<"Track (pt= "<<Tk[i].pt()<<
	  "GeV/c, eta= "<<Tk[i].eta() <<
	  ") preidentified only for track properties";
      
    

      if (GoodPreId){

	//NEW SEED with n hits
	int nHitsinSeed= min(nHitsInSeed_,Tj[i].foundHits());
	
	Trajectory seedTraj;
	OwnVector<TrackingRecHit>  rhits;
	
	vector<TrajectoryMeasurement> tm=Tj[i].measurements();

	for (int ihit=tm.size()-1; ihit>=int(tm.size()-nHitsinSeed);ihit-- ){ 
	  //for the first n measurement put the TM in the trajectory
	  // and save the corresponding hit
	  if ((*tm[ihit].recHit()).hit()->isValid()){
	    seedTraj.push(tm[ihit]);
	    rhits.push_back((*tm[ihit].recHit()).hit()->clone());
	  }
	}

	if(!seedTraj.measurements().empty()){
	  
	  PTrajectoryStateOnDet* state = TrajectoryStateTransform().
	    persistentState(seedTraj.lastMeasurement().updatedState(),
			    (*seedTraj.lastMeasurement().recHit()).hit()->geographicalId().rawId());
	  TrajectorySeed NewSeed(*state,rhits,alongMomentum);
	  
	  output_preid->push_back(NewSeed);
	  delete state;
	}
	else   output_preid->push_back(Seed);
	if(produceCkfPFT_){
	  
	  PFRecTrack pftrack( trackRef->charge(), 
				    PFRecTrack::KF_ELCAND, 
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

	  PFRecTrack pftrack( trackRef->charge(), 
				    PFRecTrack::KF, 
				    i, trackRef );

	  bool valid = pfTransformer_->addPoints( pftrack, *trackRef, Tj[i] );

	  if(valid)
	    pOutputPFRecTrackCollection->push_back(pftrack);		
	  
	}
      }
    } //end loop on track collection
  } //end loop on the vector of track collections
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
  //Magnetic Field
  ESHandle<MagneticField> magneticField;
  es.get<IdealMagneticFieldRecord>().get(magneticField);
  cout<<"GK "<<magneticField->inTesla(GlobalPoint(90,90,180))<<endl;
  B_=magneticField->inTesla(GlobalPoint(0,0,0));
  cout<<"BBB hhhh"<<B_.x()<<" "<<B_.y()<<" "<<B_.z()<<endl;
  
  pfTransformer_= new PFTrackTransformer(B_);



  
  //Resolution maps
  FileInPath ecalEtaMap(conf_.getParameter<string>("EtaMap"));
  FileInPath ecalPhiMap(conf_.getParameter<string>("PhiMap"));
  resMapEtaECAL_ = new PFResolutionMap("ECAL_eta",ecalEtaMap.fullPath().c_str());
  resMapPhiECAL_ = new PFResolutionMap("ECAL_phi",ecalPhiMap.fullPath().c_str());

  if(useTmva_){
    reader = new TMVA::Reader();
    method_ = conf_.getParameter<string>("TMVAMethod");
    
    reader->AddVariable("eP",&eP);
    reader->AddVariable("chi",&chi);
    reader->AddVariable("chired",&chired);
    reader->AddVariable("chiRatio",&chiRatio);
    reader->AddVariable("dpt",&dpt);
    reader->AddVariable("nhit",&nhit);
    reader->AddVariable("eta",&eta);
    reader->AddVariable("pt",&pt);
    FileInPath Weigths(conf_.getParameter<string>("Weights"));
    reader->BookMVA( method_, Weigths.fullPath().c_str()  );	
    }

    
    //read threshold
    FileInPath parFile(conf_.getParameter<string>("ThresholdFile"));
    ifstream ifs(parFile.fullPath().c_str());
    for (int iy=0;iy<72;iy++) ifs >> thr[iy];
    
    //read PS threshold
    FileInPath parPSFile(conf_.getParameter<string>("PSThresholdFile"));
    ifstream ifsPS(parPSFile.fullPath().c_str());
    for (int iy=0;iy<12;iy++) ifsPS >> thrPS[iy];
 
}
int GoodSeedProducer::getBin(float pt){
int ip=0;
  if (pt<6) ip=0;
  else {  if (pt<12) ip=1;
        else ip=2;
  }
return ip;
}
int GoodSeedProducer::getBin(float eta, float pt){
  int ie=0;
  int ip=0;
  if (fabs(eta)<1.2) ie=0;
  else{ if (fabs(eta)<1.68) ie=1;
    else ie=2;
  }
  if (pt<6) ip=0;
  else {  if (pt<12) ip=1;     
	else ip=2;
  }
  int iep= ie*3+ip;
  LogDebug("GoodSeedProducer")<<"Track pt ="<<pt<<" eta="<<eta<<" bin="<<iep;
  return iep;
}

void GoodSeedProducer::PSforTMVA(XYZTLorentzVector mom,XYZTLorentzVector pos ){

  BaseParticlePropagator OutParticle(RawParticle(mom,pos)
				     ,0.,0.,B_.z()) ;

  OutParticle.propagateToPreshowerLayer1(false);
  if (OutParticle.getSuccess()!=0){
    //   GlobalPoint v1=ps1TSOS.globalPosition();
    math::XYZPoint v1=math::XYZPoint(OutParticle.vertex());
    if ((v1.Rho() >=
	 PFGeometry::innerRadius(PFGeometry::PS1)) &&
	(v1.Rho() <=
	 PFGeometry::outerRadius(PFGeometry::PS1))) {
      float enPScl1=0;
      float chi1=100;
      vector<PFCluster>::const_iterator ips;
      for (ips=ps1Clus.begin(); ips!=ps1Clus.end();ips++){
	float ax=((*ips).position().x()-v1.x())/0.114;
	float ay=((*ips).position().y()-v1.y())/2.43;
	float pschi= sqrt(ax*ax+ay*ay);
	if (pschi<chi1){
	  chi1=pschi;
	  enPScl1=(*ips).energy();
	}
      }
      ps1En=enPScl1;
      ps1chi=chi1;


      OutParticle.propagateToPreshowerLayer2(false);
      if (OutParticle.getSuccess()!=0){
	math::XYZPoint v2=math::XYZPoint(OutParticle.vertex());
	if ((v2.Rho() >=
	     PFGeometry::innerRadius(PFGeometry::PS2)) &&
	    (v2.Rho() <=
	     PFGeometry::outerRadius(PFGeometry::PS2))){
	  float enPScl2=0;
	  float chi2=100;
	  for (ips=ps2Clus.begin(); ips!=ps2Clus.end();ips++){
	    float ax=((*ips).position().x()-v2.x())/1.88;
	    float ay=((*ips).position().y()-v2.y())/0.1449;
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
