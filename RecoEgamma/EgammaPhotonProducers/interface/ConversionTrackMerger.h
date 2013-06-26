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
// $Author: wmtan $
// $Date: 2011/05/20 17:17:28 $
// $Revision: 1.2 $
//

#include "FWCore/Framework/interface/EDProducer.h"
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

  class ConversionTrackMerger : public edm::EDProducer
  {
  public:

    explicit ConversionTrackMerger(const edm::ParameterSet& conf);

    virtual ~ConversionTrackMerger();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ParameterSet conf_;

    std::auto_ptr<reco::ConversionTrackCollection> outputTrks;
  };


#endif
