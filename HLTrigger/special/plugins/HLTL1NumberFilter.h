#ifndef HLTrigger_special_HLTL1NumberFilter_h
#define HLTrigger_special_HLTL1NumberFilter_h
// -*- C++ -*-
//
// Package:    HLTrigger/special
// Class:      HLTL1NumberFilter
//
/**\class HLTL1NumberFilter HLTL1NumberFilter.cc filter/HLTL1NumberFilter/src/HLTL1NumberFilter.cc

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
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <string>

//
// class declaration
//

class HLTL1NumberFilter : public edm::global::EDFilter<> {
public:
  explicit HLTL1NumberFilter(const edm::ParameterSet &);
  ~HLTL1NumberFilter() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  bool filter(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  // ----------member data ---------------------------

  /// raw data
  const edm::EDGetTokenT<FEDRawDataCollection> inputToken_;
  /// accept the event if its event number is a multiple of period_
  const unsigned int period_;
  /// FEDs from which to get lv1ID number
  const std::vector<int> fedIds_;
  /// if invert_=true, invert that event accept decision
  const bool invert_;
  /// if useTCDS=true, use 64-bit Event Number from TCDS record (FED 1024) word 11
  const bool useTCDS_;
};

#endif
