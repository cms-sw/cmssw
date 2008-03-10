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
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

using namespace std;
using namespace edm;
PFElecTkProducer::PFElecTkProducer(const ParameterSet& iConfig):
  conf_(iConfig),
  pfTransformer_(0)
{
  LogInfo("PFElecTkProducer")<<"PFElecTkProducer started";

  gsfTrackLabel_ = iConfig.getParameter<InputTag>
    ("GsfTrackModuleLabel");

  pfTrackLabel_ = iConfig.getParameter<InputTag>
    ("PFRecTrackLabel");

  produces<reco::PFRecTrackCollection>();

  trajinev_ = iConfig.getParameter<bool>("TrajInEvents");
  modemomentum_ = iConfig.getParameter<bool>("ModeMomentum");
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
  auto_ptr< reco::PFRecTrackCollection > 
    gsfPFRecTrackCollection(new reco::PFRecTrackCollection);

  
  //read collections of tracks
  Handle<reco::GsfTrackCollection> gsfelectrons;
  iEvent.getByLabel(gsfTrackLabel_,gsfelectrons);

  //read collections of trajectories
  Handle<vector<Trajectory> > TrajectoryCollection;
 
  //read pfrectrack collection
  Handle<reco::PFRecTrackCollection> thePfRecTrackCollection;
  iEvent.getByLabel(pfTrackLabel_,thePfRecTrackCollection);
  const reco::PFRecTrackCollection PfRTkColl = *(thePfRecTrackCollection.product());

  if (trajinev_){
    iEvent.getByLabel(gsfTrackLabel_,TrajectoryCollection); 
    reco::GsfTrackCollection gsftracks = *(gsfelectrons.product());
    vector<Trajectory> tjvec= *(TrajectoryCollection.product());
   
    for (uint igsf=0; igsf<gsftracks.size();igsf++) {


      reco::PFRecTrackRef kf_ref(thePfRecTrackCollection,
			      FindPfRef(PfRTkColl,gsftracks[igsf]));
      
      reco::GsfTrackRef trackRef(gsfelectrons, igsf);
      
      reco::PFRecTrack pftrack( gsftracks[igsf].charge(), 
				reco::PFRecTrack::GSF, 
				igsf, trackRef,
				kf_ref);
      
//       bool valid = pfTransformer_->addPoints( pftrack, 
// 					      gsftracks[igsf] , 
// 					      tjvec[igsf] );
      
      bool validgsfbrem = pfTransformer_->addPointsAndBrems(pftrack, 
					gsftracks[igsf], 
					tjvec[igsf],
					modemomentum_);
      
    //   if(valid)
// 	gsfPFRecTrackCollection->push_back(pftrack);
      if(validgsfbrem)
	gsfPFRecTrackCollection->push_back(pftrack);
    }
    iEvent.put(gsfPFRecTrackCollection);
  }else LogError("PFEleTkProducer")<<"No trajectory in the events";
  
}
// ------------- method for find the corresponding kf pfrectrack ---------------------
int
PFElecTkProducer::FindPfRef(const reco::PFRecTrackCollection  & PfRTkColl, 
			    reco::GsfTrack gsftk){


  if (&(*gsftk.seedRef())==0) return -1;

  reco::PFRecTrackCollection::const_iterator pft=PfRTkColl.begin();
  reco::PFRecTrackCollection::const_iterator pftend=PfRTkColl.end();
  uint i_pf=0;
  int ibest=-1;
  uint ish_max=0;
  for(;pft!=pftend;++pft){
    uint ish=0;
    if (pft->algoType()==reco::PFRecTrack::KF_ELCAND){
      
      
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
 	  if (hit->sharesInput(
 			       (*hhit)->clone() , 
			       TrackingRecHit::all )) ish++;
	  
 	}	
      }
      if (ish>ish_max){
	ish_max=ish;
	ibest=i_pf;
      }
      
    }
    
    i_pf++;
  }
  if (ibest<0) return -1;
  
  return ibest;
}

// ------------ method called once each job just before starting event loop  ------------
void 
PFElecTkProducer::beginJob(const EventSetup& iSetup)
{
  pfTransformer_= new PFTrackTransformer();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFElecTkProducer::endJob() {
}

//define this as a plug-in
