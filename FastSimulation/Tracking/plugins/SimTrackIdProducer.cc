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
  edm::InputTag trackCollectionTag = conf.getParameter<edm::InputTag>("trajectories"); 
 
  max_Chi2 = conf.getParameter<double>("maxChi2");


  if (conf.exists("overrideTrkQuals")) {
    edm::InputTag overrideTrkQuals = conf.getParameter<edm::InputTag>("overrideTrkQuals");
    if ( !(overrideTrkQuals==edm::InputTag("")) )
      overrideTrkQuals_.push_back( consumes<edm::ValueMap<int> >(overrideTrkQuals) );
  }
  trackQuality_=reco::TrackBase::undefQuality;
  filterTracks_=false;
  if (conf.exists("TrackQuality")){
    filterTracks_=true;
    std::string trackQuality = conf.getParameter<std::string>("TrackQuality");
    if ( !trackQuality.empty() ) {
      trackQuality_=reco::TrackBase::qualityByName(trackQuality);
      //   minNumberOfLayersWithMeasBeforeFiltering_ = iConfig.existsAs<int>("minNumberOfLayersWithMeasBeforeFiltering") ?                                                    
      //iConfig.getParameter<int>("minNumberOfLayersWithMeasBeforeFiltering") : 0;                                                                                            
    }
  }
    
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
  
  std::vector<edm::Handle<edm::ValueMap<int> > > quals;
  if ( overrideTrkQuals_.size() > 0) {
    quals.resize(1);
    e.getByToken(overrideTrkQuals_[0],quals[0]);
  }

  reco::TrackCollection::const_iterator aTrack = trackCollection->begin();
  reco::TrackCollection::const_iterator lastTrack = trackCollection->end();
  
  for ( ; aTrack!=lastTrack; ++aTrack)
  {
    reco::TrackRef trackRef(trackCollection,aTrack->index());
    if (filterTracks_) {
      bool goodTk = true;
      
      if ( quals.size()!=0) {
        int qual=(*(quals[0]))[trackRef];
        if ( qual < 0 ) {goodTk=false;}
        //note that this does not work for some trackquals (goodIterative or undefQuality)                 
        else
          goodTk = ( qual & (1<<trackQuality_))>>trackQuality_;
	  }
      else
        goodTk=(aTrack->quality(trackQuality_));
      if ( !goodTk) continue;    
    }
    if(aTrack->chi2()>max_Chi2) continue ; 
    
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
  e.put(SimTrackIds);  
}
