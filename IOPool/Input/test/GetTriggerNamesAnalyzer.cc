// -*- C++ -*-
//
// Package:     IOPool/Input
// Class  :     GetTriggerNamesAnalyzer
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 11 Sep 2023 13:00:39 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

namespace edmtest {
  class GetTriggerNamesAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit GetTriggerNamesAnalyzer(edm::ParameterSet const&);

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const final;

  private:
    edm::EDGetTokenT<edm::TriggerResults> const trToken_;
  };
}  // namespace edmtest

using namespace edmtest;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GetTriggerNamesAnalyzer::GetTriggerNamesAnalyzer(edm::ParameterSet const&)
    : trToken_(consumes(edm::InputTag("TriggerResults", "", edm::InputTag::kSkipCurrentProcess))) {}

//
// const member functions
//
void GetTriggerNamesAnalyzer::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
  if (iEvent.triggerNames(iEvent.get(trToken_)).triggerNames().empty()) {
    throw cms::Exception("TestFailed") << " trigger names is empty";
  }
}

DEFINE_FWK_MODULE(edmtest::GetTriggerNamesAnalyzer);
