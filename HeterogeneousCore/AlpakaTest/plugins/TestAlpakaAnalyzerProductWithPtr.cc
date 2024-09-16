#include "DataFormats/PortableTestObjects/interface/TestProductWithPtr.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

/**
 * This class is part of testing CopyToHost<T>::postCopy().
 */
class TestAlpakaAnalyzerProductWithPtr : public edm::global::EDAnalyzer<> {
public:
  TestAlpakaAnalyzerProductWithPtr(edm::ParameterSet const& iConfig)
      : token_(consumes(iConfig.getParameter<edm::InputTag>("src"))) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    descriptions.addDefault(desc);
  }

  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const override {
    auto const& prod = iEvent.get(token_);
    auto view = prod.view();
    for (int i = 0; i < view.metadata().size(); ++i) {
      auto const expected = i * 2 + 1;
      if (expected != view.ptr()[i]) {
        throw cms::Exception("Assert") << "Expected " << expected << " got " << view.ptr()[i];
      }
      if (view.buffer(i) != view.ptr()[i]) {
        throw cms::Exception("Assert") << "Buffer has " << view.buffer(i) << " via pointer " << view.ptr()[i];
      }
    }
  }

private:
  edm::EDGetTokenT<portabletest::TestProductWithPtr<alpaka_common::DevHost>> token_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestAlpakaAnalyzerProductWithPtr);
