#include <cassert>
#include <string>

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

class TestAlpakaAnalyzer : public edm::stream::EDAnalyzer<> {
public:
  TestAlpakaAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")},
        token_{consumes<portabletest::TestHostCollection>(source_)} {}

  void analyze(edm::Event const& event, edm::EventSetup const&) override {
    portabletest::TestHostCollection const& product = event.get(token_);

    for (int32_t i = 0; i < product->size(); ++i) {
      assert(product->id(i) == i);
    }

    edm::LogInfo("TestAlpakaAnalyzer") << source_.encode() << ".size() = " << product->size();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::InputTag source_;
  const edm::EDGetTokenT<portabletest::TestHostCollection> token_;
};

#include "HeterogeneousCore/AlpakaCore/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestAlpakaAnalyzer);
