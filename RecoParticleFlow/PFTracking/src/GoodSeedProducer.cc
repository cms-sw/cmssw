// -*- C++ -*-
//
// Package:    PFTracking
// Class:      GoodSeedProducer
// 
// Original Author:  Michele Pioppi

#include "RecoParticleFlow/PFTracking/interface/GoodSeedProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFResolutionMap.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"  
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include <fstream>
#include <string>

using namespace edm;
using namespace std;
reco::PFResolutionMap* GoodSeedProducer::resMapEtaECAL_ = 0;                                        
reco::PFResolutionMap* GoodSeedProducer::resMapPhiECAL_ = 0;

GoodSeedProducer::GoodSeedProducer(const ParameterSet& iConfig):
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
  
  etaresmap_=iConfig.getParameter<string>("EtaMap");
  phiresmap_=iConfig.getParameter<string>("PhiMap");

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
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GoodSeedProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
 
  std::vector<reco::PFRecTrack> pftracks;
  pftracks.clear();
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
  for (iklus=thePSPfClustCollection.product()->begin();
       iklus!=thePSPfClustCollection.product()->end();
       iklus++){
    if((*iklus).energy()>clusThreshold_) basClus.push_back(*iklus);
  }
  
  LogDebug("GoodSeedProducer")<<"Number of tracks to be analyzed "<<Tj.size();

  for(uint i=0;i<Tk.size();i++){

    float PTIN=Tj[i].firstMeasurement().updatedState().globalMomentum().perp();
    float PTFIN=Tj[i].lastMeasurement().updatedState().globalMomentum().perp();
    float PTOB=Tj[i].lastMeasurement().updatedState().globalMomentum().mag();
    float chired=Tk[i].normalizedChi2();
    int nhitpi=Tj[i].foundHits();
    TrajectorySeed Seed=Tj[i].seed();
 
    float pttin=(PTIN>0) ? fabs(PTFIN-PTIN)/PTIN : 0.;



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

    if(ecalTsos.isValid()){
      float etarec=ecalTsos.globalPosition().eta();
      float phirec=ecalTsos.globalPosition().phi();
      
      for(vector<reco::PFCluster>::const_iterator aClus = basClus.begin();
	  aClus != basClus.end(); aClus++) {
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
    
      ReferenceCountingPointer<Surface> showerMaxWall=
	pfTransformer_->showerMaxSurface(EE,true,ecalTsos,side);

      if (&(*showerMaxWall)!=0){
	TSOS maxShTsos=propagator_.product()->propagate(ecalTsos, *showerMaxWall);
	
	if (maxShTsos.isValid()){

	  toteta+=(etarec-maxShTsos.globalPosition().eta());
	  totphi+=(phirec-maxShTsos.globalPosition().phi());
	}
      }

    }
 
  
    //thresholds 
    int ibin=getBin(Tk[i].eta(),Tk[i].pt())*10;

    float chi2cut=thr[ibin+0];
    float ep_cutmin=thr[ibin+1];
    //
    int hit1max=int(thr[ibin+2]);
    float chiredmin=thr[ibin+3];
    float pttmin=thr[ibin+4];
    //
    float chiratiocut=thr[ibin+5]; 
    float gschicut=thr[ibin+6]; 
    float gsptmin=thr[ibin+7];
    // 
    int hit2max=int(thr[ibin+8]);
    float finchicut=thr[ibin+9]; 
    //



    double ecaletares 
      = resMapEtaECAL_->GetBinContent(resMapEtaECAL_->FindBin(feta,EE));
    double ecalphires 
      = resMapPhiECAL_->GetBinContent(resMapPhiECAL_->FindBin(feta,EE)); 


    float chieta= toteta/ecaletares;
    float chiphi= totphi/ecalphires;
    float chichi= chieta*chieta + chiphi*chiphi;

    //Matching criteria
    bool aa1=(chichi<chi2cut);
    bool aa2= ((EP>ep_cutmin)&&(EP<1.2));
    bool aa3= (aa1 && aa2);

    //KF filter
    bool bb1 =
      ((pttin>pttmin) || (chired>chiredmin) || (nhitpi<hit1max));

    bool bb2 = false;
    bool bb3 = false;

    if((!aa3)&&bb1){

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
	      //	      float gchi=SmooTjs[0].chiSquared()/max(1,SmooTjs[0].foundHits()-5);
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
    bool cc1=(aa3 || bb5); 
   
    if(cc1)
      LogDebug("GoodSeedProducer")<<"Track (pt= "<<Tk[i].pt()<<
	"GeV/c, eta= "<<Tk[i].eta() <<
	") preidentified for agreement between  track and ECAL cluster";
    if(cc1 &&(!bb1))
      LogDebug("GoodSeedProducer")<<"Track (pt= "<<Tk[i].pt()<<
	"GeV/c, eta= "<<Tk[i].eta() <<
	") preidentified only for track properties";


 
    if (cc1){

      //NEW SEED with n hits
      int nHitsinSeed= min(nHitsInSeed_,Tj[i].foundHits());

      Trajectory seedTraj;
      edm::OwnVector<TrackingRecHit>  rhits;

      vector<TrajectoryMeasurement> tm=Tj[i].measurements();

      for (uint ihit=tm.size()-1; ihit>=tm.size()-nHitsinSeed;ihit-- ){ 
	//for the first n measurement put the TM in the trajectory
	// and save the corresponding hit
	seedTraj.push(tm[ihit]);
	rhits.push_back((*tm[ihit].recHit()).hit()->clone());
      }
      PTrajectoryStateOnDet* state = TrajectoryStateTransform().
	persistentState(seedTraj.lastMeasurement().updatedState(),
			(*seedTraj.lastMeasurement().recHit()).hit()->geographicalId().rawId());
      
      TrajectorySeed NewSeed(*state,rhits,alongMomentum);

      output_preid->push_back(NewSeed);
      
      if(produceCkfPFT_){
	reco::PFRecTrack pft=pfTransformer_->
	  producePFtrackKf(&(Tj[i]),&(Tk[i]),reco::PFRecTrack::KF_ELCAND,i);
	pftracks.push_back(pft);

      } 
    }else{
      if (produceCkfseed_){
	output_nopre->push_back(Seed);
      }
      if(produceCkfPFT_){
	reco::PFRecTrack pft=pfTransformer_->
	  producePFtrackKf(&(Tj[i]),&(Tk[i]),reco::PFRecTrack::KF,i);
	pftracks.push_back(pft); 
      }
    }
  }
  
  if(produceCkfPFT_){
    for(uint ipf=0; ipf<pftracks.size();ipf++)
      pOutputPFRecTrackCollection->push_back(pftracks[ipf]);
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
  edm::ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);
  pfTransformer_= new PFTrackTransformer(magField.product());

  //Tracking Tools
  es.get<TrackingComponentsRecord>().get(propagatorName_, propagator_);
  es.get<TrackingComponentsRecord>().get(fitterName_, fitter_);
  es.get<TrackingComponentsRecord>().get(smootherName_, smoother_);

  //Resolution maps
  resMapEtaECAL_ = new reco::PFResolutionMap("ECAL_eta",etaresmap_.c_str());
  resMapPhiECAL_ = new reco::PFResolutionMap("ECAL_phi",phiresmap_.c_str());

  //read threshold
  std::string parfile = conf_.getParameter<string>
    ("ThresholdFile");
  std::string name = "RecoParticleFlow/PFTracking/data/";
  name+=parfile;
  edm::FileInPath parFile(name);
  std::ifstream ifs(parFile.fullPath().c_str());
  for (int iy=0;iy<150;iy++) ifs >> thr[iy];

}

int GoodSeedProducer::getBin(float eta, float pt){
  int ie=0;
  int ip=0;
  if (fabs(eta)<0.8) ie=0;
  else{ if (fabs(eta)<1.6) ie=1;
    else ie=2;
  }
  if (pt<2) ip=0;
  else {  if (pt<4) ip=1;
    else {  if (pt<6) ip=2;
      else {  if (pt<9) ip=3;
	else ip=4;
      }
    }
  }
  int iep= ie*5+ip;
  LogDebug("GoodSeedProducer")<<"Track pt ="<<pt<<" eta="<<eta<<" bin="<<iep;
  return iep;
}
