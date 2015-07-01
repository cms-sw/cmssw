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

SimTrackIdProducer::SimTrackIdProducer(const edm::ParameterSet& conf)
{
  //Main products
  produces<std::vector<unsigned int> >();

  // Input Tag
  edm::InputTag trackCollectionTag = conf.getParameter<edm::InputTag>("trackCollection"); 
 
  maxChi2_ = conf.getParameter<double>("maxChi2");

  auto const & classifier = conf.getParameter<edm::InputTag>("trackClassifier");
  if ( !classifier.label().empty())
     srcQuals = consumes<QualityMaskCollection>(classifier);


  trackQuality_=reco::TrackBase::undefQuality;
  filterTracks_=false;
  if (conf.exists("TrackQuality")){
    filterTracks_=true;
    std::string trackQuality = conf.getParameter<std::string>("TrackQuality");
    if ( !trackQuality.empty() ) {
      trackQuality_=reco::TrackBase::qualityByName(trackQuality);
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
 
  unsigned char qualMask = ~0;
  if (trackQuality_!=reco::TrackBase::undefQuality) qualMask = 1<<trackQuality_; 

  QualityMaskCollection const * pquals=nullptr;
  if (!srcQuals.isUninitialized()) {
     edm::Handle<QualityMaskCollection> hqual;
     e.getByToken(srcQuals, hqual);
     pquals = hqual.product();
  }

  for (size_t i = 0 ; i!=trackCollection->size();++i)
  {
    const reco::Track & track = (*trackCollection)[i];
    if (filterTracks_) {
      bool goodTk =  (pquals) ? (*pquals)[i] & qualMask : track.quality(trackQuality_);
      if ( !goodTk) continue;
    }
    if(track.chi2()>maxChi2_) continue ; 
    
    const TrackingRecHit * hit = *(track.recHitsBegin());
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
