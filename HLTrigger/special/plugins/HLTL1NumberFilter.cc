// -*- C++ -*-
//
// Package:    HLTrigger/special
// Class:      HLTL1NumberFilter
//
/**\class HLTL1NumberFilter HLTL1NumberFilter.cc HLTrigger/special/plugins/HLTL1NumberFilter.cc

Description:

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Martin Grunewald
//         Created:  Tue Jan 22 13:55:00 CET 2008
//
//

// system include files
#include <string>
#include <iostream>
#include <memory>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/TCDS/interface/TCDSRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTL1NumberFilter.h"

//
// constructors and destructor
//
HLTL1NumberFilter::HLTL1NumberFilter(const edm::ParameterSet& config)
    :  //now do what ever initialization is needed
      inputToken_(consumes<FEDRawDataCollection>(config.getParameter<edm::InputTag>("rawInput"))),
      period_(config.getParameter<unsigned int>("period")),
      fedIds_(config.getParameter<std::vector<int>>("fedIds")),
      invert_(config.getParameter<bool>("invert")),
      // only try and use TCDS event number if the FED ID 1024 OR 1050 is selected
      useTCDS_(config.getParameter<bool>("useTCDSEventNumber") &&
               std::any_of(fedIds_.begin(), fedIds_.end(), [](int id) { return id == 1024 || id == 1050; })) {}

void HLTL1NumberFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("rawInput", edm::InputTag("source"));
  desc.add<unsigned int>("period", 4096);
  desc.add<bool>("invert", true);
  desc.add<std::vector<int>>("fedIds", {812});
  desc.add<bool>("useTCDSEventNumber", false);
  descriptions.add("hltL1NumberFilter", desc);
}
//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTL1NumberFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  if (iEvent.isRealData()) {
    edm::Handle<FEDRawDataCollection> theRaw;
    iEvent.getByToken(inputToken_, theRaw);

    bool accept = false;
    bool atLeastOneValidFED = false;  // Track if any FED had valid data
    for (int fedId : fedIds_) {
      const FEDRawData& data = theRaw->FEDData(fedId);
      if (data.data() && data.size() > 0) {
        atLeastOneValidFED = true;
        unsigned long counter;
        // Use TCDS if requested and fedId is 1024 or 1050
        if (useTCDS_ && (fedId == 1024 || fedId == 1050)) {
          TCDSRecord record(data.data());
          counter = record.getTriggerCount();
        } else {
          FEDHeader header(data.data());
          counter = header.lvl1ID();
        }
        if (period_ != 0)
          accept = accept || (counter % period_ == 0);
        else
          accept = true;
      }
    }
    if (!atLeastOneValidFED) {
      edm::LogWarning("HLTL1NumberFilter") << "No valid data for any FED in list used by HLTL1NumberFilter";
    }
    if (invert_)
      accept = !accept;
    return accept;
  } else {
    return true;
  }
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTL1NumberFilter);
