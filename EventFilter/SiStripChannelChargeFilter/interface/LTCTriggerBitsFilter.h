#ifndef LTCTriggerBitsFilter_H
#define LTCTriggerBitsFilter_H 

// -*- C++ -*-
//
// Package:     SiStripChannelChargeFilter
// Class  :     TrackMTCCFilter
// 
//
// Original Author:  dkcira

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <boost/cstdint.hpp>

namespace cms
{
 class LTCTriggerBitsFilter : public edm::EDFilter {
  public:
    LTCTriggerBitsFilter(const edm::ParameterSet& ps);
    virtual ~LTCTriggerBitsFilter() {}
    virtual bool filter(edm::Event & e, edm::EventSetup const& c);
  private:
    std::vector<uint32_t> AcceptFromBits;
  };
}
#endif
