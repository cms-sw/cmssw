#ifndef HLTHcalNZSFilter_h
#define HLTHcalNZSFilter_h
// -*- C++ -*-
//
// Package:    HLTHcalNZSFilter
// Class:      HLTHcalNZSFilter
//
/**\class HLTHcalNZSFilter HLTHcalNZSFilter.cc filter/HLTHcalNZSFilter/src/HLTHcalNZSFilter.cc

Description: Filter to select HCAL non-ZS events

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Bryan DAHMES
//         Created:  Tue Jan 22 13:55:00 CET 2008
//
//


// include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <string>

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTHcalNZSFilter : public HLTFilter {
public:
  explicit HLTHcalNZSFilter(const edm::ParameterSet&);
  ~HLTHcalNZSFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<FEDRawDataCollection> dataInputToken_;
  edm::InputTag dataInputTag_;

};

#endif
