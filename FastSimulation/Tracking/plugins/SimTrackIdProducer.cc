// -*- C++ -*-
//
// Package:    FastSimulation/SimTrackIdProducer
// Class:      SimTrackIdProducer
// 
/**\class SimTrackIdProducer SimTrackIdProducer.cc FastSimulation/SimTrackIdProducer/plugins/SimTrackIdProducer.cc

 Description: the class finds Ids of SimTracks by looping over all reco tracks, looking for a recHit in it, reading out the Id of the track, and storing it in SimTrackIds vector.

*/
//
// Original Author:  Vilius Kripas
//         Created:  Fri, 24 Oct 2014 09:47:25 GMT
//
//
#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FastSimulation/Tracking/plugins/SimTrackIdProducer.h"
#include <vector>
#include <stdio.h> 

SimTrackIdProducer::SimTrackIdProducer(const edm::ParameterSet& conf)
{
  //Main products
  produces<std::vector<unsigned int> >();

  // Input Tag
  edm::InputTag trackCollectionTag = conf.getParameter<edm::InputTag>("trackCollection"); 

  // consumes
  trackToken = consumes<reco::TrackCollection>(trackCollectionTag); 
}

void
SimTrackIdProducer::produce(edm::Event& e, const edm::EventSetup& es)
{     
  // The produced object
  std::auto_ptr<std::vector<unsigned int> > SimTrackIds(new std::vector<unsigned int>());
  
  // The input track collection handle
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByToken(trackToken,trackCollection);
   
  reco::TrackCollection::const_iterator aTrack = trackCollection->begin();
  reco::TrackCollection::const_iterator lastTrack = trackCollection->end();
  
  for ( ; aTrack!=lastTrack; ++aTrack)
  {
    if((aTrack->quality(aTrack->qualityByName("highPurity")))&&(aTrack->chi2()<9)){
      const TrackingRecHit* hit = *aTrack->recHitsBegin();
      if (hit)
      {
          const SiTrackerGSMatchedRecHit2D* fsimhit = dynamic_cast<const SiTrackerGSMatchedRecHit2D*>(hit);
          if (fsimhit)
          {
              SimTrackIds->push_back(fsimhit->simtrackId());
          }
      }
    }
  }
  e.put(SimTrackIds);  
}
