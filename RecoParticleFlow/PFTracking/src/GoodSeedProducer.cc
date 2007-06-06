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
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include <fstream>
#include <string>

using namespace edm;
using namespace std;
using namespace reco;
PFResolutionMap* GoodSeedProducer::resMapEtaECAL_ = 0;                                        
PFResolutionMap* GoodSeedProducer::resMapPhiECAL_ = 0;

GoodSeedProducer::GoodSeedProducer(const ParameterSet& iConfig):
  maxShPropagator_(0),
  pfTransformer_(0),
  conf_(iConfig)
{
  LogInfo("GoodSeedProducer")<<"Electron PreIdentification started  ";

  //now do what ever initialization is needed
 
  refitLabel_= 
    iConfig.getParameter<InputTag>("RefitModuleLabel");

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



  Handle<vector<Trajectory> > tjCollection;
  iEvent.getByLabel(refitLabel_, tjCollection);
  vector<Trajectory> Tj=*(tjCollection.product());

  Handle<reco::TrackCollection> tkRefCollection;
  iEvent.getByLabel(refitLabel_, tkRefCollection);
  reco::TrackCollection  Tk=*(tkRefCollection.product());

			      
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
<<<<<<< GoodSeedProducer.cc

  
  for (iklus=thePSPfClustCollection.product()->begin();
       iklus!=thePSPfClustCollection.product()->end();
       iklus++){
    //layer==-11 first layer of PS
    //layer==-12 secon layer of PS
    if ((*iklus).layer()==-11) ps1Clus.push_back(*iklus);
    if ((*iklus).layer()==-12) ps2Clus.push_back(*iklus);
  }

=======
>>>>>>> 1.14
  
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


  LogDebug("GoodSeedProducer")<<"Number of tracks to be analyzed "<<Tj.size();


  InputTag tPartTag =conf_.getParameter<InputTag>("TrackParticleTag");
  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  iEvent.getByLabel(tPartTag,TPCollectionH);
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());
 
  gTrack=0;
  gPs1=0;
  gPs2=0;


  vector< pair<uint,int> > vec_index;
  reco::SimToRecoCollection q = 
    associatorByHits->associateSimToReco(tkRefCollection,TPCollectionH,&iEvent );
  for(SimTrackContainer::size_type i=0; i<tPC.size(); ++i){
    TrackingParticleRef tp (TPCollectionH,i);
    if (tp->genParticle().size()>0){
      try{ 
	std::vector<std::pair<TrackRef, double> > trackV = q[tp];

	for (std::vector<std::pair<TrackRef,double> >::const_iterator it=trackV.begin(); it != trackV.end(); ++it) {
	  TrackRef tr = it->first;

	  if (it==trackV.begin())  vec_index.push_back(make_pair(tr.index(),tp->pdgId()));
	} 
      } catch (Exception event) {
	
      }
    }
  }
  

  for(uint i=0;i<Tk.size();i++){

    int particle_code=0;

    for (uint ivec=0; ivec<vec_index.size();ivec++){

      if (vec_index[ivec].first==i)particle_code=vec_index[ivec].second;
      
    }


    float P2=Tj[i].lastMeasurement().updatedState().globalMomentum().perp();
    float P1=Tj[i].firstMeasurement().updatedState().globalMomentum().perp();
    float ptin= (P1>0)? fabs((P2/P1)-1):0;

    float PTOB=Tj[i].lastMeasurement().updatedState().globalMomentum().mag();
    float chired=Tk[i].normalizedChi2();
    int nhitpi=Tj[i].foundHits();
    TrajectorySeed Seed=Tj[i].seed();
 

    //CLUSTERS - TRACK matching

    int side=0;  
    TSOS ecalTsos=
      pfTransformer_->getStateOnSurface(PFGeometry::ECALInnerWall,
				       Tj[i].firstMeasurement().updatedState(),
				       propagator_.product(),side);  
 

    float toteta=1000;
    float totphi=1000;
    float dr=1000;
    float EP=900;
    float EE=0;
    float feta=0;
    float fphi=0;
    //    float CorrEn=0;
    // float nclps=0;
    float etarecbest=0;
    float phirecbest=0;
    float el_dpt=0;
    float gchi=0;
    float chiratio=0;
    //    float CorrEn= PSCorrEnergy(Tj[i].firstMeasurement().updatedState()).first;
    int nclps= PSCorrEnergy(Tj[i].firstMeasurement().updatedState()).second;
    nclps+=2; 
   if(ecalTsos.isValid()){
      float   etarec=ecalTsos.globalPosition().eta();
      float  phirec=ecalTsos.globalPosition().phi();
      
      

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
	  fphi= aClus->positionXYZ().phi();
	  phirecbest=phirec;
	  etarecbest=etarec;
	}
      }
    }
 
  



    double ecaletares 
      = resMapEtaECAL_->GetBinContent(resMapEtaECAL_->FindBin(feta,EE));
    double ecalphires 
      = resMapPhiECAL_->GetBinContent(resMapPhiECAL_->FindBin(feta,EE)); 

<<<<<<< GoodSeedProducer.cc
=======
    float chieta= toteta/ecaletares;
    float chiphi= totphi/ecalphires;
    float chichi= chieta*chieta + chiphi*chiphi;

    //Matching criteria
    bool aa1= (chichi<chi2cut);
    bool aa2= ((EP>ep_cutmin)&&(EP<1.2));
    bool aa3= (aa1 && aa2);
    int ipsbin=(ibin>=90)? 4*((ibin/9)-10) :-1;
    bool aa4= ((aa3)||(ibin<90))? false : PSCorrEnergy(Tj[i].firstMeasurement().updatedState(),ipsbin);
    bool aa5= (aa3 || aa4);
    //KF filter
    bool bb1 =
      ((chired>chiredmin) || (nhitpi<hit1max));

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
	      float gchi=chiratio*chired;

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
    bool cc1=(aa5 || bb5); 
   
    if(cc1)
      LogDebug("GoodSeedProducer")<<"Track (pt= "<<Tk[i].pt()<<
	"GeV/c, eta= "<<Tk[i].eta() <<
	") preidentified for agreement between  track and ECAL cluster";
    if(cc1 &&(!bb1))
      LogDebug("GoodSeedProducer")<<"Track (pt= "<<Tk[i].pt()<<
	"GeV/c, eta= "<<Tk[i].eta() <<
	") preidentified only for track properties";


>>>>>>> 1.14
 




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
	    el_dpt=(pt_in>0) ? fabs(pt_out-pt_in)/pt_in : 0.;
	    chiratio=SmooTjs[0].chiSquared()/Tj[i].chiSquared();
	    gchi=chiratio*chired;
	    
	    //Criteria based on electron tracks
	    
	  }
	}
      }
    }

    gCode[gTrack]=particle_code;
    gEta[gTrack]=Tk[i].eta();
    gPhi[gTrack]=Tk[i].phi();
    gPt[gTrack]=Tk[i].pt();
    gDpt[gTrack]=ptin;
    gAbsPFin[gTrack]=PTOB;
    gNhit[gTrack]=nhitpi;
    gChired[gTrack]=chired;
    gPropPhi[gTrack]=phirecbest;
    gPropEta[gTrack]=etarecbest;
    gResPhi[gTrack]=ecalphires;
    gResEta[gTrack]=ecaletares;
    gClE[gTrack]=EE;
    gClPhi[gTrack]=fphi;
    gClEta[gTrack]=feta;
    //  gPSCorr[gTrack]=CorrEn;
    //   gPSNCl[gTrack]=nclps;
    gsfDpt[gTrack]=el_dpt;
    gsfChired[gTrack]=gchi;
    gsfChiRatio[gTrack]=chiratio;
    gTrack++;    
  }
  

   t1->Fill();
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

  //read threshold
  FileInPath parFile(conf_.getParameter<string>("ThresholdFile"));
  ifstream ifs(parFile.fullPath().c_str());
  for (int iy=0;iy<135;iy++) ifs >> thr[iy];
<<<<<<< GoodSeedProducer.cc
  edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
  es.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",theHitsAssociator);
  associatorByHits = (TrackAssociatorBase *) theHitsAssociator.product();
 
  outputfile = conf_.getParameter<std::string>("OutputFile");
  hFile=new TFile(outputfile.c_str(), "RECREATE");
  t1 = new TTree("t1","Reconst ntuple");
  t1->Branch("gTrack",&gTrack,"gTrack/I");
  t1->Branch("gCode",gCode,"gCode[gTrack]/F");
  t1->Branch("gEta",gEta,"gEta[gTrack]/F");
  t1->Branch("gPhi",gPhi,"gPhi[gTrack]/F");
  t1->Branch("gPt",gPt,"gPt[gTrack]/F");
  t1->Branch("gDpt",gDpt,"gDpt[gTrack]/F");
  t1->Branch("gAbsPFin",gAbsPFin,"gAbsPFin[gTrack]/F");
  t1->Branch("gNhit",gNhit,"gNhit[gTrack]/F");
  t1->Branch("gChired",gChired,"gChired[gTrack]/F");
  t1->Branch("gPropPhi",gPropPhi,"gPropPhi[gTrack]/F");
  t1->Branch("gPropEta",gPropEta,"gPropEta[gTrack]/F");
  t1->Branch("gResPhi",gResPhi,"gResPhi[gTrack]/F");
  t1->Branch("gResEta",gResEta,"gResEta[gTrack]/F");
  t1->Branch("gClE",gClE,"gClE[gTrack]/F");
  t1->Branch("gClPhi",gClPhi,"gClPhi[gTrack]/F");
  t1->Branch("gClEta",gClEta,"gClEta[gTrack]/F");
  t1->Branch("gsfDpt",gsfDpt,"gsfDpt[gTrack]/F");
  t1->Branch("gsfChired",gsfChired,"gsfChired[gTrack]/F");
  t1->Branch("gsfChiRatio",gsfChiRatio,"gsfChiRatio[gTrack]/F");
  t1->Branch("gPs1",&gPs1,"gPs1/I");
  t1->Branch("gPs1_tk",gPs1_tk,"gPs1_tk[gPs1]/I");
  t1->Branch("gPs1_dx",gPs1_dx,"gPs1_dx[gPs1]/F");
  t1->Branch("gPs1_dy",gPs1_dy,"gPs1_dy[gPs1]/F");
  t1->Branch("gPs1_en",gPs1_en,"gPs1_en[gPs1]/F");
  t1->Branch("gPs2",&gPs2,"gPs2/I");
  t1->Branch("gPs2_tk",gPs2_tk,"gPs2_tk[gPs2]/I");
  t1->Branch("gPs2_dx",gPs2_dx,"gPs2_dx[gPs2]/F");
  t1->Branch("gPs2_dy",gPs2_dy,"gPs2_dy[gPs2]/F");
  t1->Branch("gPs2_en",gPs2_en,"gPs2_en[gPs2]/F");
=======

  //read PS threshold
  FileInPath parPSFile(conf_.getParameter<string>("PSThresholdFile"));
  ifstream ifsPS(parPSFile.fullPath().c_str());
  for (int iy=0;iy<20;iy++) ifsPS >> thrPS[iy];

>>>>>>> 1.14
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
<<<<<<< GoodSeedProducer.cc
void
GoodSeedProducer::endJob() {
  hFile->Write();
  hFile->Close();
}

pair<float,int>
GoodSeedProducer::PSCorrEnergy(const TSOS tsos){
  uint iPScl=0;
  int sder=0;
  float corrEn=0;
  TSOS ps1TSOS =
    pfTransformer_->getStateOnSurface(PFGeometry::PS1Wall, tsos,
                                      propagator_.product(), sder);
  if (!(ps1TSOS.isValid())) return make_pair(0.,iPScl);
  GlobalPoint v1=ps1TSOS.globalPosition();
  
  if (!((v1.perp() >=
         PFGeometry::innerRadius(PFGeometry::PS1)) &&
        (v1.perp() <=
         PFGeometry::outerRadius(PFGeometry::PS1)))) return make_pair(0.,iPScl);
 
  
  vector<PFCluster>::const_iterator ips;
  for (ips=ps1Clus.begin(); ips!=ps1Clus.end();ips++){
    if ((fabs(v1.x()-(*ips).positionXYZ().x())<0.6)&&
        (fabs(v1.y()-(*ips).positionXYZ().y())<7)&&
        (v1.z()*(*ips).positionXYZ().z()>0)) {
      //      cout<<"PS1 "<<(*ips).energy()<<" "<<(*ips).positionXYZ().x()<<" "<<(*ips).positionXYZ().y()<<" "<<
      //        (*ips).positionXYZ().z()<<" "<<
      //      fabs(v1.x()-(*ips).positionXYZ().x())<<" "<<fabs(v1.y()-(*ips).positionXYZ().y())<<endl;
      //      corrEn+=(2*(*ips).energy());
      gPs1_tk[gPs1]=gTrack;
      gPs1_dx[gPs1]=v1.x()-(*ips).positionXYZ().x();
      gPs1_dy[gPs1]=v1.y()-(*ips).positionXYZ().y();
      gPs1_en[gPs1]=(*ips).energy();
      gPs1++;
      iPScl++;
    }
  }
  TSOS ps2TSOS =
    pfTransformer_->getStateOnSurface(PFGeometry::PS2Wall, ps1TSOS,
                                      propagator_.product(), sder);
  if (!(ps2TSOS.isValid())) return make_pair(corrEn,iPScl);
  GlobalPoint v2=ps2TSOS.globalPosition();
  //  cout<<"xxxPS2xxx "<<v2<<endl;
  if (!((v2.perp() >=
         PFGeometry::innerRadius(PFGeometry::PS2)) &&
        (v2.perp() <=
         PFGeometry::outerRadius(PFGeometry::PS2)))) return make_pair(0.,iPScl);
 
  for (ips=ps2Clus.begin(); ips!=ps2Clus.end();ips++){
    if ((fabs(v2.x()-(*ips).positionXYZ().x())<10.)&&
        (fabs(v2.y()-(*ips).positionXYZ().y())<0.9)&&
        (v2.z()*(*ips).positionXYZ().z()>0)) {

      gPs2_tk[gPs2]=gTrack;
      gPs2_dx[gPs2]=v2.x()-(*ips).positionXYZ().x();
      gPs2_dy[gPs2]=v2.y()-(*ips).positionXYZ().y();
      gPs2_en[gPs2]=(*ips).energy();
      gPs2++;
 //      cout<<"PS2 "<<(*ips).energy()<<" "<<(*ips).positionXYZ().x()<<" "<<(*ips).positionXYZ().y()<<" "<<
//         (*ips).positionXYZ().z()<<" "<<
//         fabs(v2.x()-(*ips).positionXYZ().x())<<" "<<fabs(v2.y()-(*ips).positionXYZ().y())<<endl;
      corrEn+=(3*(*ips).energy());
      iPScl+=100;
    }
  }
  
  return make_pair(corrEn,iPScl);
   
}
=======

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
>>>>>>> 1.14
