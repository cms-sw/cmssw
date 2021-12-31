#ifndef HLTEventNumberFilter_h
#define HLTEventNumberFilter_h
// -*- C++ -*-
//
// Package:    HLTEventNumberFilter
// Class:      HLTEventNumberFilter
//
/**\class HLTEventNumberFilter HLTEventNumberFilter.cc filter/HLTEventNumberFilter/src/HLTEventNumberFilter.cc

Description: Filter to select HCAL abort gap events

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Martin Grunewald
//         Created:  Tue Jan 22 13:55:00 CET 2008
//
//

// include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include <string>

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTEventNumberFilter : public edm::global::EDFilter<> {
public:
  explicit HLTEventNumberFilter(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  /// accept the event if its event number is a multiple of period_
  unsigned int period_;
  /// if invert_=true, invert that event accept decision
  bool invert_;
};

#endif
