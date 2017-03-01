#ifndef HLTHcalCalibTypeFilter_h
#define HLTHcalCalibTypeFilter_h
// -*- C++ -*-
//
// Package:    HLTHcalCalibTypeFilter
// Class:      HLTHcalCalibTypeFilter
// 
/**\class HLTHcalCalibTypeFilter HLTHcalCalibTypeFilter.cc filter/HLTHcalCalibTypeFilter/src/HLTHcalCalibTypeFilter.cc

Description: Filter to select HCAL abort gap events

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
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <string>
#include <array>
#include <atomic>

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTHcalCalibTypeFilter : public edm::global::EDFilter<> {
public:
  explicit HLTHcalCalibTypeFilter(const edm::ParameterSet&);
  virtual ~HLTHcalCalibTypeFilter();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
private:
  virtual bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  virtual void endJob(void) override;
  
  // ----------member data ---------------------------
  const edm::EDGetTokenT<FEDRawDataCollection> DataInputToken_;
  const std::vector<int> CalibTypes_;
  const bool Summary_;
  mutable std::array<std::atomic<int>, 8> eventsByType_;
};

#endif
