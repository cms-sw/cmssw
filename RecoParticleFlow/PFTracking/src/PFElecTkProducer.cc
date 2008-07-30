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
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

using namespace std;
using namespace edm;
PFElecTkProducer::PFElecTkProducer(const ParameterSet& iConfig):
  conf_(iConfig),
  pfTransformer_(0)
{
  LogInfo("PFElecTkProducer")<<"PFElecTkProducer started";
  gsfTrackModule_ = iConfig.getParameter<string>
    ("GsfTrackModuleLabel");

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
  iEvent.getByLabel(gsfTrackModule_,gsfelectrons);

  //read collections of trajectories
  Handle<vector<Trajectory> > TrajectoryCollection;
 

  if (trajinev_){
    iEvent.getByLabel(gsfTrackModule_,TrajectoryCollection); 
    reco::GsfTrackCollection gsftracks = *(gsfelectrons.product());
    vector<Trajectory> tjvec= *(TrajectoryCollection.product());
   
    for (uint igsf=0; igsf<gsftracks.size();igsf++) {
      
      reco::TrackRef dummyRef;

      reco::PFRecTrack pftrack( gsftracks[igsf].charge(), 
				reco::PFRecTrack::GSF, 
				igsf, dummyRef );
      
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
