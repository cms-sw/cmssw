#include <cassert>

#include "DataFormats/PortableTestObjects/interface/TestHostObject.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TestAlpakaObjectAnalyzer : public edm::global::EDAnalyzer<> {
public:
  TestAlpakaObjectAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")}, token_{consumes(source_)} {
    if (std::string const& eb = config.getParameter<std::string>("expectBackend"); not eb.empty()) {
      expectBackend_ = cms::alpakatools::toBackend(eb);
      backendToken_ = consumes(edm::InputTag(source_.label(), "backend", source_.process()));
    }
  }

  void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const&) const override {
    portabletest::TestHostObject const& product = event.get(token_);

    auto const& value = product.value();
    {
      edm::LogInfo msg("TestAlpakaObjectAnalyzer");
      msg << source_.encode() << ".data() at " << product.data() << '\n';
      msg << source_.encode() << ".buffer().data() at " << product.buffer().data() << '\n';
      msg << source_.encode() << ".value() = {\n";
      msg << "  .x:  " << value.x << '\n';
      msg << "  .y:  " << value.y << '\n';
      msg << "  .z:  " << value.z << '\n';
      msg << "  .id: " << value.id << '\n';
      msg << "}\n";
    }

    // check that the product data is held in the product buffer
    assert(product.buffer().data() == product.data());

    // check that the product content is as expected
    assert(value.x == 5.);
    assert(value.y == 12.);
    assert(value.z == 13.);
    assert(value.id == 42);

    // check that the backend is as expected
    if (expectBackend_) {
      auto backend = static_cast<cms::alpakatools::Backend>(event.get(backendToken_));
      if (expectBackend_ != backend) {
        throw cms::Exception("Assert") << "Expected input backend " << cms::alpakatools::toString(*expectBackend_)
                                       << ", got " << cms::alpakatools::toString(backend);
      }
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
    desc.add<std::string>("expectBackend", "")
        ->setComment(
            "Expected backend of the input collection. Empty value means to not perform the check. Default: empty "
            "string");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::InputTag source_;
  const edm::EDGetTokenT<portabletest::TestHostObject> token_;
  edm::EDGetTokenT<unsigned short> backendToken_;
  std::optional<cms::alpakatools::Backend> expectBackend_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestAlpakaObjectAnalyzer);
