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
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

using namespace std;
using namespace edm;
PFElecTkProducer::PFElecTkProducer(const ParameterSet& iConfig):conf_(iConfig)
{
  LogInfo("PFElecTkProducer")<<"PFElecTkProducer started";
  gsfTrackModule_ = iConfig.getParameter<string>
    ("GsfTrackModuleLabel");

  produces<reco::PFRecTrackCollection>();

  trajinev_ = iConfig.getParameter<bool>("TrajInEvents");

}


PFElecTkProducer::~PFElecTkProducer()
{
 

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
      
      
//       gsfPFRecTrackCollection->
// 	push_back(pfTransformer_-> producePFtrack( &(tjvec[igsf]), 
// 						   gsftracks[igsf],
// 						   reco::PFRecTrack::GSF,
// 						   igsf));    
    }
    iEvent.put(gsfPFRecTrackCollection);
  }else LogError("PFEleTkProducer")<<"No trajectory in the events";
      
}

// ------------ method called once each job just before starting event loop  ------------
void 
PFElecTkProducer::beginJob(const EventSetup& iSetup)
{
  edm::ESHandle<MagneticField> magField;
  iSetup.get<IdealMagneticFieldRecord>().get(magField);
  pfTransformer_= new PFTrackTransformer(magField.product());
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFElecTkProducer::endJob() {
}

//define this as a plug-in
