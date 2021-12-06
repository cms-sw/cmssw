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

// system include files
#include <string>
#include <iostream>
#include <memory>

// CMSSW include files
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/AMC13Header.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUHTRData.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTHcalCalibTypeFilter : public edm::global::EDFilter<> {
public:
  explicit HLTHcalCalibTypeFilter(const edm::ParameterSet&);
  ~HLTHcalCalibTypeFilter() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<FEDRawDataCollection> inputToken_;
  const std::vector<int> calibTypes_;
};

//
// constructors and destructor
//
HLTHcalCalibTypeFilter::HLTHcalCalibTypeFilter(const edm::ParameterSet& config)
    : inputToken_(consumes<FEDRawDataCollection>(config.getParameter<edm::InputTag>("InputTag"))),
      calibTypes_(config.getParameter<std::vector<int> >("CalibTypes")) {}

void HLTHcalCalibTypeFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputTag", edm::InputTag("source"));
  desc.add<std::vector<int> >("CalibTypes", {1, 2, 3, 4, 5});
  descriptions.add("hltHcalCalibTypeFilter", desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTHcalCalibTypeFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto const& rawdata = iEvent.get(inputToken_);

  // some inits
  int numZeroes = 0, numPositives = 0;

  // loop over all HCAL FEDs
  for (int fed = FEDNumbering::MINHCALFEDID; fed <= FEDNumbering::MAXHCALuTCAFEDID; fed++) {
    // skip FEDs in between VME and uTCA
    if (fed > FEDNumbering::MAXHCALFEDID && fed < FEDNumbering::MINHCALuTCAFEDID)
      continue;

    // get raw data and check if there are empty feds
    const FEDRawData& fedData = rawdata.FEDData(fed);
    if (fedData.size() < 24)
      continue;

    if (fed <= FEDNumbering::MAXHCALFEDID) {
      // VME get event type
      int eventtype = ((const HcalDCCHeader*)(fedData.data()))->getCalibType();
      if (eventtype == 0)
        numZeroes++;
      else
        numPositives++;
    } else {
      // UTCA
      hcal::AMC13Header const* hamc13 = (hcal::AMC13Header const*)fedData.data();
      for (int iamc = 0; iamc < hamc13->NAMC(); iamc++) {
        HcalUHTRData uhtr(hamc13->AMCPayload(iamc), hamc13->AMCSize(iamc));
        int eventtype = uhtr.getEventType();
        if (eventtype == 0)
          numZeroes++;
        else
          numPositives++;
      }
    }
  }

  // if there are FEDs with Non-Collision event type, check what the majority is
  // if calibs - true
  // if 0s - false
  if (numPositives > 0) {
    if (numPositives > numZeroes)
      return true;
    else
      edm::LogWarning("HLTHcalCalibTypeFilter") << "Conflicting Calibration Types found";
  }

  // return false if there are no positives
  // and if the majority has 0 calib type
  return false;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTHcalCalibTypeFilter);
