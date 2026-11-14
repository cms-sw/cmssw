#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

#include "DataFormats/VertexSoA/interface/VertexGNNHostCollection.h"

#include <utility>

namespace edmtest {

  class TestWriteHostVertexGNNSoA : public edm::global::EDProducer<> {
  public:
    TestWriteHostVertexGNNSoA(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    unsigned int trackSize_;
    edm::EDPutTokenT<vertexgnn::TrackFeaturesHostCollection> featuresToken_;
    edm::EDPutTokenT<vertexgnn::GNNOutputHostCollection> outputToken_;
    edm::EDPutTokenT<vertexgnn::GNNSlotHostCollection> slotToken_;
  };

  TestWriteHostVertexGNNSoA::TestWriteHostVertexGNNSoA(edm::ParameterSet const& iPSet)
      : trackSize_(iPSet.getParameter<unsigned int>("trackSize")),
        featuresToken_(produces()),
        outputToken_(produces()),
        slotToken_(produces()) {}

  void TestWriteHostVertexGNNSoA::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    vertexgnn::TrackFeaturesHostCollection features(cms::alpakatools::host(), int(trackSize_));
    auto featuresView = features.view();
    for (unsigned int i = 0; i < trackSize_; ++i) {
      featuresView[i].vz() = float(i);
      featuresView[i].pt() = 0.5f * float(i);
      featuresView[i].has_time() = float(i % 2);
    }

    vertexgnn::GNNOutputHostCollection output(cms::alpakatools::host(), int(trackSize_));
    auto outputView = output.view();
    for (unsigned int i = 0; i < trackSize_; ++i) {
      for (int k = 0; k < vertexgnn::kNumSlots; ++k) {
        outputView[i].A()[k] = float(i + k);
      }
      for (int m = 0; m < 3; ++m) {
        outputView[i].pi()[m] = float(m);
      }
    }

    vertexgnn::GNNSlotHostCollection slots(cms::alpakatools::host(), vertexgnn::kNumSlots);
    auto slotView = slots.view();
    for (int k = 0; k < vertexgnn::kNumSlots; ++k) {
      slotView[k].z_hat() = float(k);
      slotView[k].t_hat() = 0.1f * float(k);
      slotView[k].p() = 1.f / float(k + 1);
    }

    iEvent.emplace(featuresToken_, std::move(features));
    iEvent.emplace(outputToken_, std::move(output));
    iEvent.emplace(slotToken_, std::move(slots));
  }

  void TestWriteHostVertexGNNSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<unsigned int>("trackSize", 500);
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteHostVertexGNNSoA;
DEFINE_FWK_MODULE(TestWriteHostVertexGNNSoA);
