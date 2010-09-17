//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           ConversionTrackProducer
// 
// Description:     Trivial producer of ConversionTrack collection from an edm::View of a track collection
//                  (ConversionTrack is a simple wrappper class containing a TrackBaseRef and some additional flags)
//
// Original Author: J.Bendavid
//
// $Author: stenson $
// $Date: 2010/05/03 23:47:08 $
// $Revision: 1.26 $
//

#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionTrackProducer.h"
#include "DataFormats/Common/interface/Handle.h"


#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

    
  ConversionTrackProducer::ConversionTrackProducer(edm::ParameterSet const& conf) : 
    conf_(conf)
  {
    produces<reco::ConversionTrackCollection>();
   
  }


  // Virtual destructor needed.
  ConversionTrackProducer::~ConversionTrackProducer() { }  

  // Functions that gets called by framework every event
  void ConversionTrackProducer::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input TrackCollection(s)
    std::string trackProducer = conf_.getParameter<std::string>("TrackProducer");
    bool setTrackerOnly = conf_.getParameter<bool>("setTrackerOnly");
    bool setArbitratedEcalSeeded = conf_.getParameter<bool>("setArbitratedEcalSeeded");    
    bool setArbitratedMerged = conf_.getParameter<bool>("setArbitratedMerged");
    
    //get input collection (through edm::View)
    edm::Handle<edm::View<reco::Track> > hTrks;
    e.getByLabel(trackProducer, hTrks);

    // Step B: create empty output collection
    outputTrks = std::auto_ptr<reco::ConversionTrackCollection>(new reco::ConversionTrackCollection);    
    
    // Simple conversion of tracks to conversion tracks, setting appropriate flags from configuration
    for (edm::RefToBaseVector<reco::Track>::const_iterator it = hTrks->refVector().begin(); it != hTrks->refVector().end(); ++it) {
      reco::ConversionTrack convTrack(*it);
      convTrack.setIsTrackerOnly(setTrackerOnly);
      convTrack.setIsArbitratedEcalSeeded(setArbitratedEcalSeeded);
      convTrack.setIsArbitratedMerged(setArbitratedMerged);
      
      outputTrks->push_back(convTrack);
    }
    
    e.put(outputTrks);
    return;

  }//end produce
