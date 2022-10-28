#ifndef TrackMTCCFilter_H
#define TrackMTCCFilter_H

// -*- C++ -*-
//
// Package:     SiStripChannelChargeFilter
// Class  :     TrackMTCCFilter
//
//
// Original Author:  dkcira

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms {
  class TrackMTCCFilter : public edm::stream::EDFilter<> {
  public:
    TrackMTCCFilter(const edm::ParameterSet& ps);
    ~TrackMTCCFilter() override {}
    bool filter(edm::Event& e, edm::EventSetup const& c) override;

  private:
    std::string TrackProducer;
    std::string TrackLabel;
    unsigned int MinNrOfTracks;
  };
}  // namespace cms
#endif
