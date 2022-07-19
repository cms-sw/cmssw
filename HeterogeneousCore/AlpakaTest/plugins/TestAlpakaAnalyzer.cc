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

class TestAlpakaAnalyzer : public edm::stream::EDAnalyzer<> {
public:
  TestAlpakaAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")}, token_{consumes(source_)} {}

  void analyze(edm::Event const& event, edm::EventSetup const&) override {
    portabletest::TestHostCollection const& product = event.get(token_);

    auto const& view = product.const_view();
    for (int32_t i = 0; i < view.metadata().size(); ++i) {
      assert(view[i].id() == i);
    }

    edm::LogInfo msg("TestAlpakaAnalyzer");
    msg << source_.encode() << ".size() = " << view.metadata().size() << '\n';
    msg << "data = " << product.buffer().data() << " x = " << view.x() << " y = " << view.y() << " z = " << view.z()
        << " id = " << view.id() << '\n';
    msg << std::hex << "[y - x] = 0x" << reinterpret_cast<intptr_t>(view.y()) - reinterpret_cast<intptr_t>(view.x())
        << " [z - y] = 0x" << reinterpret_cast<intptr_t>(view.z()) - reinterpret_cast<intptr_t>(view.y())
        << " [id - z] = 0x" << reinterpret_cast<intptr_t>(view.id()) - reinterpret_cast<intptr_t>(view.z());
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

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestAlpakaAnalyzer);
