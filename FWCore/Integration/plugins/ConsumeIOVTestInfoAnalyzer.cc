// -*- C++ -*-
//
// Package:    FWCore/Integration
// Class:      ConsumeIOVTestInfoAnalyzer
//
/**\class edmtest::ConsumeIOVTestInfoAnalyzer

 Description: Used in tests. Declares it consumes products
 of type IOVTestInfo. The purpose is to cause the ESProducer
 that produces that product to run.
*/
// Original Author:  W. David Dagenhart
//         Created:  2 January 2025

#include "GadgetRcd.h"

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Integration/interface/IOVTestInfo.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edmtest {

  class ConsumeIOVTestInfoAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit ConsumeIOVTestInfoAnalyzer(edm::ParameterSet const&);

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    edm::ESGetToken<IOVTestInfo, GadgetRcd> const esToken_;
  };

  ConsumeIOVTestInfoAnalyzer::ConsumeIOVTestInfoAnalyzer(edm::ParameterSet const& pset)
      : esToken_{esConsumes(pset.getUntrackedParameter<edm::ESInputTag>("esInputTag"))} {}

  void ConsumeIOVTestInfoAnalyzer::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {}

  void ConsumeIOVTestInfoAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<edm::ESInputTag>("esInputTag", edm::ESInputTag("", ""));
    descriptions.addDefault(desc);
  }

}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_MODULE(ConsumeIOVTestInfoAnalyzer);
