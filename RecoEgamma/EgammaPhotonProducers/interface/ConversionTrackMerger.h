#ifndef ConversionTrackMerger_h
#define ConversionTrackMerger_h

//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           ConversionTrackMerger
// 
// Description:     Hit Dumper
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
//

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "DataFormats/EgammaTrackReco/interface/ConversionTrack.h"
#include "DataFormats/EgammaTrackReco/interface/ConversionTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

  class ConversionTrackMerger : public edm::stream::EDProducer<>
  {
  public:

    explicit ConversionTrackMerger(const edm::ParameterSet& conf);

    virtual ~ConversionTrackMerger();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ParameterSet conf_;

    edm::EDGetTokenT<reco::ConversionTrackCollection> trackProducer1;
    edm::EDGetTokenT<reco::ConversionTrackCollection> trackProducer2;

    std::auto_ptr<reco::ConversionTrackCollection> outputTrks;
  };


#endif
