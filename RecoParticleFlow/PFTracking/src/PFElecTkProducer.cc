// -*- C++ -*-
//
// Package:    PFElecTkProducer
// Class:      PFElecTkProducer
// 
// Original Author:  Michele Pioppi
//         Created:  Tue Jan 23 15:26:39 CET 2007



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "RecoParticleFlow/PFTracking/interface/PFElecTkProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
//#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
using namespace std;
using namespace edm;
PFElecTkProducer::PFElecTkProducer(const ParameterSet& iConfig):conf_(iConfig),
								     trackAlgo_(iConfig)
{
  LogInfo("PFElecTkProducer")<<"PFElecTkProducer started";
  gsfTrackModule_ = iConfig.getParameter<string>
    ("GsfTrackModuleLabel");
  gsfTrackCandidateModule_
    = iConfig.getParameter<string>
    ("GsfTrackCandidateModuleLabel");
  produces<reco::PFRecTrackCollection>();

  trajinev_ = iConfig.getParameter<bool>("TrajInEvents");
  propagatorName_ = iConfig.getParameter<string>("Propagator");
  builderName_ = iConfig.getParameter<string>("TTRHBuilder"); 
  fitterName_ = iConfig.getParameter<string>("Fitter");
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
  AlgoProductCollection algoResults;
  LogDebug("PFElecTkProducer")<<"START event: "<<iEvent.id().event()
			      <<" in run "<<iEvent.id().run();

  pftracks.clear();
  //create the empty collections 
  auto_ptr< reco::PFRecTrackCollection > 
    gsfPFRecTrackCollection(new reco::PFRecTrackCollection);
  
  //read collections
  Handle<reco::GsfTrackCollection> gsfelectrons;
  iEvent.getByLabel(gsfTrackModule_,gsfelectrons);

  //   Handle<TrackCandidateCollection> gsfcandidates;
  //   iEvent.getByLabel(gsfTrackCandidateModule_,gsfcandidates);

  Handle<vector<Trajectory> > TrajectoryCollection;
  iEvent.getByLabel(gsfTrackModule_,TrajectoryCollection);
  
  if (trajinev_){
    reco::GsfTrackCollection gsftracks = *(gsfelectrons.product());
    vector<Trajectory> tjvec= *(TrajectoryCollection.product());

    for (uint igsf=0; igsf<gsftracks.size();igsf++)
      pftracks.push_back(PFTransformer->
			 producePFtrackKf(&(tjvec[igsf]),&(gsftracks[igsf]),
					  reco::PFRecTrack::GSF,igsf));
    
    
    for(uint ipf=0; ipf<pftracks.size();ipf++)
      gsfPFRecTrackCollection->push_back(pftracks[ipf]);
    iEvent.put(gsfPFRecTrackCollection);
  }else LogError("PFEleTkProducer")<<"No trajectory in the events";
      
}

// ------------ method called once each job just before starting event loop  ------------
void 
PFElecTkProducer::beginJob(const EventSetup& iSetup)
{
 
    iSetup.get<IdealMagneticFieldRecord>().get(theMF);
    magField = theMF.product();
    iSetup.get<TrackerDigiGeometryRecord>().get(theG);
    iSetup.get<TrackingComponentsRecord>().get(propagatorName_, thePropagator);
    iSetup.get<TrackingComponentsRecord>().get(fitterName_, theFitter);
    iSetup.get<TransientRecHitRecord>().get(builderName_, theBuilder);
    PFTransformer= new PFTrackTransformer(magField);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFElecTkProducer::endJob() {
}

//define this as a plug-in
