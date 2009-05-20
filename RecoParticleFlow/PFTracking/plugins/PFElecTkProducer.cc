// -*- C++ -*-
//
// Package:    PFTracking
// Class:      PFElecTkProducer
// 
// Original Author:  Michele Pioppi
//         Created:  Tue Jan 23 15:26:39 CET 2007



// system include files
#include <memory>

// user include files
#include "RecoParticleFlow/PFTracking/interface/PFElecTkProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"



#include "TMath.h"
using namespace std;
using namespace edm;
using namespace reco;
PFElecTkProducer::PFElecTkProducer(const ParameterSet& iConfig):
  conf_(iConfig),
  pfTransformer_(0)
{
  LogInfo("PFElecTkProducer")<<"PFElecTkProducer started";

  gsfTrackLabel_ = iConfig.getParameter<InputTag>
    ("GsfTrackModuleLabel");

  pfTrackLabel_ = iConfig.getParameter<InputTag>
    ("PFRecTrackLabel");

  produces<GsfPFRecTrackCollection>();

  trajinev_ = iConfig.getParameter<bool>("TrajInEvents");
  modemomentum_ = iConfig.getParameter<bool>("ModeMomentum");
  applySel_ = iConfig.getParameter<bool>("applyEGSelection");
  applyGsfClean_ = iConfig.getParameter<bool>("applyGsfTrackCleaning");
  useFifthStep_ = iConfig.getParameter<bool>("useFifthTrackingStep");
  detaGsfSC_ = iConfig.getParameter<double>("MinDEtaGsfSC");
  dphiGsfSC_ = iConfig.getParameter<double>("MinDPhiGsfSC");
  SCEne_ = iConfig.getParameter<double>("MinSCEnergy");
}


PFElecTkProducer::~PFElecTkProducer()
{
 
  delete pfTransformer_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PFElecTkProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  LogDebug("PFElecTkProducer")<<"START event: "<<iEvent.id().event()
			      <<" in run "<<iEvent.id().run();

  //create the empty collections 
  auto_ptr< GsfPFRecTrackCollection > 
    gsfPFRecTrackCollection(new GsfPFRecTrackCollection);

  
  //read collections of tracks
  Handle<GsfTrackCollection> gsfelectrons;
  iEvent.getByLabel(gsfTrackLabel_,gsfelectrons);

  //read collections of trajectories
  Handle<vector<Trajectory> > TrajectoryCollection;
 
  //read pfrectrack collection
  Handle<PFRecTrackCollection> thePfRecTrackCollection;
  iEvent.getByLabel(pfTrackLabel_,thePfRecTrackCollection);
  const PFRecTrackCollection PfRTkColl = *(thePfRecTrackCollection.product());

  if (trajinev_){
    iEvent.getByLabel(gsfTrackLabel_,TrajectoryCollection); 
    GsfTrackCollection gsftracks = *(gsfelectrons.product());
    vector<Trajectory> tjvec= *(TrajectoryCollection.product());
  
    for (uint igsf=0; igsf<gsftracks.size();igsf++) {
      
      GsfTrackRef trackRef(gsfelectrons, igsf);
     
      int kf_ind=FindPfRef(PfRTkColl,gsftracks[igsf],false);
      
      if (kf_ind>=0) {
	
	PFRecTrackRef kf_ref(thePfRecTrackCollection,
			     kf_ind);
	
        
	if(useFifthStep_ == false) {
	  TrackRef kfref = kf_ref->trackRef();
	  unsigned int Algo = kfref->algo() < 5 ? kfref->algo()-1 : kfref->algo()-5;
	  if ( Algo >= 4 ) {
	    continue;
	  }
	}

	pftrack_=GsfPFRecTrack( gsftracks[igsf].charge(), 
				reco::PFRecTrack::GSF, 
				igsf, trackRef,
				kf_ref);
      } else  {
	PFRecTrackRef dummyRef;
	pftrack_=GsfPFRecTrack( gsftracks[igsf].charge(), 
				reco::PFRecTrack::GSF, 
				igsf, trackRef,
				dummyRef);
      }
      bool validgsfbrem = pfTransformer_->addPointsAndBrems(pftrack_, 
							    gsftracks[igsf], 
							    tjvec[igsf],
							    modemomentum_);
      bool passSel = true;
      bool keepGsf = true;
      if(applySel_) 
	passSel = applySelection(gsftracks[igsf]);      

      if(applyGsfClean_) 
	keepGsf = resolveGsfTracks(gsftracks,igsf);
      

      if(validgsfbrem && passSel && keepGsf)
	gsfPFRecTrackCollection->push_back(pftrack_);
    }
    //OTHER GSF TRACK COLLECTION
    if(conf_.getParameter<bool>("AddGSFTkColl")){
     
      Handle<GsfElectronCollection> ElecCollection;
      iEvent.getByLabel(conf_.getParameter<InputTag >("GsfElectrons"), ElecCollection);
      GsfElectronCollection::const_iterator iel = ElecCollection->begin();
      GsfElectronCollection::const_iterator iel_end = ElecCollection->end();

      Handle<GsfTrackCollection> otherGsfColl;
      iEvent.getByLabel(conf_.getParameter<InputTag >("GsfTracks"),otherGsfColl);
      GsfTrackCollection othergsftracks = *(otherGsfColl.product());

      Handle<vector<Trajectory> > TrajectoryCollection;
      iEvent.getByLabel(conf_.getParameter<InputTag >("GsfTracks"),TrajectoryCollection);
      vector<Trajectory> newtj= *(TrajectoryCollection.product());
 
      for(;iel!=iel_end;++iel){
       uint ibest =9999; float diffbest=10000.;
       for (uint igsf=0; igsf<othergsftracks.size();igsf++) {
	 float diff =(iel->gsfTrack()->momentum()-othergsftracks[igsf].momentum()).Mag2();
	 if (diff<diffbest){
	   ibest=igsf;
	   diffbest=diff;
	 }
       }

       if (ibest==9999 || diffbest>0.00001) continue;

       if(otherElId(gsftracks,othergsftracks[ibest])){
	 GsfTrackRef trackRef(otherGsfColl, ibest);
	 
	 int kf_ind=FindPfRef(PfRTkColl,othergsftracks[ibest],true);
	 
	 if (kf_ind>=0) {	      
	   PFRecTrackRef kf_ref(thePfRecTrackCollection,
				kf_ind);
	   pftrack_=GsfPFRecTrack( othergsftracks[ibest].charge(), 
				   reco::PFRecTrack::GSF, 
				   ibest, trackRef,
				   kf_ref);
	 } else  {
	   PFRecTrackRef dummyRef;
	   pftrack_=GsfPFRecTrack( othergsftracks[ibest].charge(), 
				   reco::PFRecTrack::GSF, 
				   ibest, trackRef,
				   dummyRef);
	 }
	 bool validgsfbrem = pfTransformer_->addPointsAndBrems(pftrack_, 
							       othergsftracks[ibest], 
							       newtj[ibest],
							       modemomentum_); 
	 if(validgsfbrem)
	   gsfPFRecTrackCollection->push_back(pftrack_);
	 
       }
      }
    }
    




    iEvent.put(gsfPFRecTrackCollection);
  }else LogError("PFEleTkProducer")<<"No trajectory in the events";
  
}
// ------------- method for find the corresponding kf pfrectrack ---------------------
int
PFElecTkProducer::FindPfRef(const reco::PFRecTrackCollection  & PfRTkColl, 
			    reco::GsfTrack gsftk,
			    bool otherColl){


  if (&(*gsftk.seedRef())==0) return -1;
  ElectronSeedRef ElSeedRef=gsftk.extra()->seedRef().castTo<ElectronSeedRef>();
  //CASE 1 ELECTRONSEED DOES NOT HAVE A REF TO THE CKFTRACK
  if (ElSeedRef->ctfTrack().isNull()){
    reco::PFRecTrackCollection::const_iterator pft=PfRTkColl.begin();
    reco::PFRecTrackCollection::const_iterator pftend=PfRTkColl.end();
    uint i_pf=0;
    int ibest=-1;
    uint ish_max=0;
    float dr_min=1000;
    //SEARCH THE PFRECTRACK THAT SHARES HITS WITH THE ELECTRON SEED
    for(;pft!=pftend;++pft){
      uint ish=0;
      
      float dph= fabs(pft->trackRef()->phi()-gsftk.phi()); 
      if (dph>TMath::Pi()) dph-= TMath::TwoPi();
      float det=fabs(pft->trackRef()->eta()-gsftk.eta());
      float dr =sqrt(dph*dph+det*det);  
      
      trackingRecHit_iterator  hhit=
	pft->trackRef()->recHitsBegin();
      trackingRecHit_iterator  hhit_end=
	pft->trackRef()->recHitsEnd();
      
    
      
      for(;hhit!=hhit_end;++hhit){
	if (!(*hhit)->isValid()) continue;
	TrajectorySeed::const_iterator hit=
	  gsftk.seedRef()->recHits().first;
	TrajectorySeed::const_iterator hit_end=
	  gsftk.seedRef()->recHits().second;
 	for(;hit!=hit_end;++hit){
	  if (!(hit->isValid())) continue;

	  if((hit->geographicalId()==(*hhit)->geographicalId())&&
	     (((*hhit)->localPosition()-hit->localPosition()).mag()<0.01)) ish++;
 	}	
	
      }
      

      if ((ish>ish_max)||
	  ((ish==ish_max)&&(dr<dr_min))){
	ish_max=ish;
	dr_min=dr;
	ibest=i_pf;
      }
      
   
    
      i_pf++;
    }
    if (ibest<0) return -1;
    
    if((ish_max==0) || (dr_min>0.05))return -1;
    if(otherColl && (ish_max==0)) return -1;
    return ibest;
  }
  else{
    //ELECTRON SEED HAS A REFERENCE
   
    reco::PFRecTrackCollection::const_iterator pft=PfRTkColl.begin();
    reco::PFRecTrackCollection::const_iterator pftend=PfRTkColl.end();
    uint i_pf=0;
    
    for(;pft!=pftend;++pft){
      //REF COMPARISON
      if (pft->trackRef()==ElSeedRef->ctfTrack()){
	return i_pf;
      }
      i_pf++;
    }
  }
  return -1;
}

bool 
PFElecTkProducer::otherElId(const reco::GsfTrackCollection  & GsfColl, 
			    reco::GsfTrack GsfTk){
  int nhits=GsfTk.numberOfValidHits();
  GsfTrackCollection::const_iterator igs=GsfColl.begin();
  GsfTrackCollection::const_iterator igs_end=GsfColl.end();  
  uint shared=0;
  for(;igs!=igs_end;++igs){
    uint tmp_sh=0;
    trackingRecHit_iterator  ghit=igs->recHitsBegin();
    trackingRecHit_iterator  ghit_end=igs->recHitsEnd();
    for (;ghit!=ghit_end;++ghit){

      if (!((*ghit)->isValid())) continue;
      
      trackingRecHit_iterator  hit=GsfTk.recHitsBegin();
      trackingRecHit_iterator  hit_end=GsfTk.recHitsEnd();
 
      for (;hit!=hit_end;++hit){
	if (!((*hit)->isValid())) continue;
	if(((*hit)->geographicalId()==(*ghit)->geographicalId())&&
	   (((*hit)->localPosition()-(*ghit)->localPosition()).mag()<0.01)) tmp_sh++;
      }
    }
    if (tmp_sh>shared) shared=tmp_sh;
  }
  return ((float(shared)/float(nhits))<0.5);
}
// -- method to apply gsf electron selection to EcalDriven seeds
bool 
PFElecTkProducer::applySelection(reco::GsfTrack gsftk) {
  if (&(*gsftk.seedRef())==0) return false;
  ElectronSeedRef ElSeedRef=gsftk.extra()->seedRef().castTo<ElectronSeedRef>();

  bool passCut = false;
  if (ElSeedRef->ctfTrack().isNull()){
    if(ElSeedRef->caloCluster().isNull()) return passCut;
    SuperClusterRef scRef = ElSeedRef->caloCluster().castTo<SuperClusterRef>();
    //do this just to know if exist a SC? 
    if(scRef.isNonnull()) {
      float caloEne = scRef->energy();
      float feta = fabs(scRef->eta()-gsftk.etaMode());
      float fphi = fabs(scRef->phi()-gsftk.phiMode());
      if (fphi>TMath::Pi()) fphi-= TMath::TwoPi();
      if(caloEne > SCEne_ && feta < detaGsfSC_ && fabs(fphi) < dphiGsfSC_)
	passCut = true;
    }
  }
  else {
    // get all the gsf found by tracker driven
    passCut = true;
  }
  return passCut;
}
bool 
PFElecTkProducer::resolveGsfTracks(const reco::GsfTrackCollection  & GsfCol, unsigned int ngsf) {
  if (&(*GsfCol[ngsf].seedRef())==0) return false;    
  ElectronSeedRef nElSeedRef=GsfCol[ngsf].extra()->seedRef().castTo<ElectronSeedRef>();
  

  bool n_keepGsf = true;
  const math::XYZPoint nxyz = GsfCol[ngsf].innerPosition();
  int nhits=GsfCol[ngsf].numberOfValidHits();
  int ncharge = GsfCol[ngsf].chargeMode();
  TrajectoryStateOnSurface outTSOS = mtsTransform_.outerStateOnSurface(GsfCol[ngsf]);
  TrajectoryStateOnSurface inTSOS = mtsTransform_.innerStateOnSurface(GsfCol[ngsf]);
  int outCharge = -2;
  int inCharge = -2;
  if(outTSOS.isValid())
    outCharge = mtsMode_->chargeFromMode(outTSOS);	  
  if(inTSOS.isValid())
    inCharge = mtsMode_->chargeFromMode(inTSOS);
  

  float nchi2 = GsfCol[ngsf].chi2();
  float neta = GsfCol[ngsf].etaMode();
  float nphi = GsfCol[ngsf].phiMode();
  float ndist = sqrt(nxyz.x()*nxyz.x()+
		     nxyz.y()*nxyz.y()+
		     nxyz.z()*nxyz.z());
  
  
  
  for (uint igsf=0; igsf<GsfCol.size();igsf++) {
    if(igsf != ngsf ) {
      
      float ieta = GsfCol[igsf].etaMode();
      float iphi = GsfCol[igsf].phiMode();
      float feta = fabs(neta - ieta);
      float fphi = fabs(nphi - iphi);
      if (fphi>TMath::Pi()) fphi-= TMath::TwoPi();     


      if(feta < 0.05 && fabs(fphi) < 0.3) {

	TrajectoryStateOnSurface i_outTSOS = mtsTransform_.outerStateOnSurface(GsfCol[igsf]);
	TrajectoryStateOnSurface i_inTSOS = mtsTransform_.innerStateOnSurface(GsfCol[igsf]);
	int i_outCharge = -2;
	int i_inCharge = -2;
	if(i_outTSOS.isValid())
	  i_outCharge = mtsMode_->chargeFromMode(i_outTSOS);	  
	if(i_inTSOS.isValid())
	  i_inCharge = mtsMode_->chargeFromMode(i_inTSOS);
	

	if (&(*GsfCol[igsf].seedRef())==0) continue;    
	ElectronSeedRef iElSeedRef=GsfCol[igsf].extra()->seedRef().castTo<ElectronSeedRef>();

	// First Case: both gsf track have a reference to a SC: cleaning using SC 
	if(nElSeedRef->caloCluster().isNonnull() && iElSeedRef->caloCluster().isNonnull()) {

	  SuperClusterRef nscRef = nElSeedRef->caloCluster().castTo<SuperClusterRef>();
	  if(nscRef.isNull()) {
	    n_keepGsf = false;
	    return n_keepGsf;
	  }    
	  float nEP = nscRef->energy()/GsfCol[ngsf].pMode();
	  SuperClusterRef iscRef = iElSeedRef->caloCluster().castTo<SuperClusterRef>();
	  if(iscRef.isNonnull()) {
	    if(nscRef == iscRef) {
	      float iEP =  iscRef->energy()/GsfCol[igsf].pMode();
	     
	      
	      trackingRecHit_iterator  nhit=GsfCol[ngsf].recHitsBegin();
	      trackingRecHit_iterator  nhit_end=GsfCol[ngsf].recHitsEnd();
	      unsigned int tmp_sh = 0;
	      for (;nhit!=nhit_end;++nhit){
		if ((*nhit)->isValid()){
		  trackingRecHit_iterator  ihit=GsfCol[igsf].recHitsBegin();
		  trackingRecHit_iterator  ihit_end=GsfCol[igsf].recHitsEnd();
		  for (;ihit!=ihit_end;++ihit){
		    if ((*ihit)->isValid()) {
		      if((*nhit)->sharesInput(&*(*ihit),TrackingRecHit::all))  tmp_sh++; 
		    }
		  }
		}
	      }
	      if (tmp_sh>0) {
		if(fabs(iEP-1) < fabs(nEP-1) 
		   && i_outCharge == i_inCharge
		   && i_outCharge != -2) {
		  n_keepGsf = false;
		  return n_keepGsf;
		}
		if(outCharge != inCharge) {
		  n_keepGsf = false;
		  return n_keepGsf;
		}
	      }			      
	    }
	  }
	}
	else {
	  // Second Case: One Gsf has reference to a SC and the other one not or both not
	  // Cleaning using: starting point 
	  const math::XYZPoint ixyz = GsfCol[igsf].innerPosition();
	  float idist = sqrt(ixyz.x()*ixyz.x()+
			     ixyz.y()*ixyz.y()+
			     ixyz.z()*ixyz.z());
	  int ihits=GsfCol[igsf].numberOfValidHits();
	  float ichi2 = GsfCol[igsf].chi2();
	  int icharge = GsfCol[igsf].chargeMode();
	  
	  if (idist < (ndist-5)) {
	    n_keepGsf = false;
	    return n_keepGsf;
	  }
	  else if(ndist > (idist-5)){
	    // Thirt Case:  One Gsf has reference to a SC and the other one not or both not
	    // gsf tracks starts from the same layer
	    // check number of sharing modules (at least 50%)
	    // check number of sharing hits (at least 2)
	    // check charge flip inner/outer
	    
	    unsigned int sharedMod = 0;
	    unsigned int sharedHits = 0;
	    
	    trackingRecHit_iterator  nhit=GsfCol[ngsf].recHitsBegin();
	    trackingRecHit_iterator  nhit_end=GsfCol[ngsf].recHitsEnd();
	    for (;nhit!=nhit_end;++nhit){
	      if ((*nhit)->isValid()){
		trackingRecHit_iterator  ihit=GsfCol[igsf].recHitsBegin();
		trackingRecHit_iterator  ihit_end=GsfCol[igsf].recHitsEnd();
		for (;ihit!=ihit_end;++ihit){
		  if ((*ihit)->isValid()) {
		    if((*ihit)->geographicalId()==(*nhit)->geographicalId()) sharedMod++;
		    if((*nhit)->sharesInput(&*(*ihit),TrackingRecHit::all))  sharedHits++; 
		  }
		}
	      }
	    }
	    unsigned int den = ihits;
	    if(nhits < ihits)
	      den = nhits;
	    float fracMod = sharedMod*1./den*1.;
	    
	    TrajectoryStateOnSurface i_outTSOS = mtsTransform_.outerStateOnSurface(GsfCol[igsf]);
	    TrajectoryStateOnSurface i_inTSOS = mtsTransform_.innerStateOnSurface(GsfCol[igsf]);
	    int i_outCharge = -2;
	    int i_inCharge = -2;
	    if(i_outTSOS.isValid())
	      i_outCharge = mtsMode_->chargeFromMode(i_outTSOS);	  
	    if(i_inTSOS.isValid())
	    i_inCharge = mtsMode_->chargeFromMode(i_inTSOS);
	    
	    
	    if(fracMod > 0.5 && sharedHits > 1 && icharge == ncharge && i_outCharge == i_inCharge) {
	      if(ihits > nhits) {
		n_keepGsf = false;
		return n_keepGsf;
	      }
	      else if(ihits == nhits  && ichi2 < nchi2) {
		n_keepGsf = false;
		return n_keepGsf;
	      }
	    }
	    if(fracMod > 0.3 && sharedHits > 1 && outCharge != -2 && inCharge != outCharge) {
	      n_keepGsf = false;
	      return n_keepGsf;
	    }
	  }
	}
      }
    }
  }
  return n_keepGsf;
}
// ------------ method called once each job just before starting event loop  ------------
void 
PFElecTkProducer::beginRun(edm::Run& run,
			   const EventSetup& iSetup)
{
  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

  ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);


  mtsTransform_ = MultiTrajectoryStateTransform(tracker.product(),magneticField.product());
  

  pfTransformer_= new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0,0,0))));

  

}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFElecTkProducer::endRun() {
  delete pfTransformer_;
}

//define this as a plug-in
