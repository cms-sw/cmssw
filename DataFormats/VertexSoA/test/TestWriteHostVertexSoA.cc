#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/VertexSoA/interface/ZVertexHost.h"

#include <memory>
#include <utility>
#include <vector>

namespace edmtest {

  class TestWriteHostVertexSoA : public edm::global::EDProducer<> {
  public:
    TestWriteHostVertexSoA(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    unsigned int vertexSize_;
    edm::EDPutTokenT<ZVertexHost> putToken_;
  };

  TestWriteHostVertexSoA::TestWriteHostVertexSoA(edm::ParameterSet const& iPSet)
      : vertexSize_(iPSet.getParameter<unsigned int>("vertexSize")), putToken_(produces()) {}

  void TestWriteHostVertexSoA::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    ZVertexHost ZVertexs({{int(vertexSize_), int(4 * vertexSize_)}}, cms::alpakatools::host());
    auto ZVertexsView = ZVertexs.view();
    for (unsigned int i = 0; i < vertexSize_; ++i) {
      ZVertexsView[i].chi2() = float(i);
    }
    iEvent.emplace(putToken_, std::move(ZVertexs));
  }

  void TestWriteHostVertexSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<unsigned int>("vertexSize", 1000);
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteHostVertexSoA;
DEFINE_FWK_MODULE(TestWriteHostVertexSoA);
