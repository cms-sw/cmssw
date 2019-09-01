// -*- C++ -*-
//
// Package:     SiStripChannelChargeFilter
// Class  :     TrackMTCCFilter
//
//
// Original Author:  dkcira

#include "EventFilter/SiStripChannelChargeFilter/interface/TrackMTCCFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace cms {

  TrackMTCCFilter::TrackMTCCFilter(const edm::ParameterSet& ps) {
    TrackProducer = ps.getParameter<std::string>("TrackProducer");
    TrackLabel = ps.getParameter<std::string>("TrackLabel");
    MinNrOfTracks = ps.getParameter<int>("MinNrOfTracks");
    produces<int>();
    edm::LogInfo("TrackMTCCFilter") << "TrackProducer = " << TrackProducer;
    edm::LogInfo("TrackMTCCFilter") << "TrackLabel = " << TrackLabel;
    edm::LogInfo("TrackMTCCFilter") << "MinNrOfTracks = " << MinNrOfTracks;
  }

  bool TrackMTCCFilter::filter(edm::Event& e, edm::EventSetup const& c) {
    bool decision = false;  // default value, only accept if set true in this loop

    //get SiStripCluster
    edm::Handle<reco::TrackCollection> trackCollection;
    e.getByLabel(TrackProducer, TrackLabel, trackCollection);

    unsigned int nroftracks = trackCollection->size();
    //  edm::LogInfo("TrackMTCCFilter")<<"trackCollection->size()="<<nroftracks;
    if (nroftracks >= MinNrOfTracks)
      decision = true;

    e.put(std::make_unique<int>(decision));
    return decision;
  }

}  // namespace cms
