#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/VertexSoA/interface/VertexGNNHostCollection.h"

namespace edmtest {

  class TestReadHostVertexGNNSoA : public edm::global::EDAnalyzer<> {
  public:
    TestReadHostVertexGNNSoA(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    edm::EDGetTokenT<vertexgnn::TrackFeaturesHostCollection> featuresToken_;
    edm::EDGetTokenT<vertexgnn::GNNOutputHostCollection> outputToken_;
    edm::EDGetTokenT<vertexgnn::GNNSlotHostCollection> slotToken_;
  };

  TestReadHostVertexGNNSoA::TestReadHostVertexGNNSoA(edm::ParameterSet const& iPSet)
      : featuresToken_(consumes(iPSet.getParameter<edm::InputTag>("input"))),
        outputToken_(consumes(iPSet.getParameter<edm::InputTag>("input"))),
        slotToken_(consumes(iPSet.getParameter<edm::InputTag>("input"))) {}

  void TestReadHostVertexGNNSoA::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& features = iEvent.get(featuresToken_);
    auto const& output = iEvent.get(outputToken_);
    auto const& slots = iEvent.get(slotToken_);
    auto featuresView = features.const_view();
    auto outputView = output.const_view();
    auto slotView = slots.const_view();

    if (featuresView.metadata().size() == 0 || featuresView.metadata().size() != outputView.metadata().size()) {
      throw cms::Exception("TestReadHostVertexGNNSoA Failure")
          << "TestReadHostVertexGNNSoA::analyze, size mismatch " << featuresView.metadata().size() << " "
          << outputView.metadata().size();
    }
    if (slotView.metadata().size() != vertexgnn::kNumSlots) {
      throw cms::Exception("TestReadHostVertexGNNSoA Failure")
          << "TestReadHostVertexGNNSoA::analyze, slot size " << slotView.metadata().size();
    }
    for (int i = 0; i < featuresView.metadata().size(); ++i) {
      if (featuresView[i].vz() != float(i) || featuresView[i].pt() != 0.5f * float(i) ||
          featuresView[i].has_time() != float(i % 2)) {
        throw cms::Exception("TestReadHostVertexGNNSoA Failure")
            << "TestReadHostVertexGNNSoA::analyze, features entry, i = " << i;
      }
      for (int k = 0; k < vertexgnn::kNumSlots; ++k) {
        if (outputView[i].A()[k] != float(i + k)) {
          throw cms::Exception("TestReadHostVertexGNNSoA Failure")
              << "TestReadHostVertexGNNSoA::analyze, output entry, i = " << i << " k = " << k;
        }
      }
      for (int m = 0; m < 3; ++m) {
        if (outputView[i].pi()[m] != float(m)) {
          throw cms::Exception("TestReadHostVertexGNNSoA Failure")
              << "TestReadHostVertexGNNSoA::analyze, pi entry, i = " << i << " m = " << m;
        }
      }
    }
    for (int k = 0; k < vertexgnn::kNumSlots; ++k) {
      if (slotView[k].z_hat() != float(k) || slotView[k].t_hat() != 0.1f * float(k) ||
          slotView[k].p() != 1.f / float(k + 1)) {
        throw cms::Exception("TestReadHostVertexGNNSoA Failure")
            << "TestReadHostVertexGNNSoA::analyze, slot entry, k = " << k;
      }
    }
  }

  void TestReadHostVertexGNNSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("input");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadHostVertexGNNSoA;
DEFINE_FWK_MODULE(TestReadHostVertexGNNSoA);
