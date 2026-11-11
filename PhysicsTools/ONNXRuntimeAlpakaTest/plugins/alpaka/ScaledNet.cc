#include "DataFormats/PortableTestObjects/interface/alpaka/HitDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/SimpleNetDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/alpaka/AlpakaSession.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest {

  using cms::Ort::alpakatools::TensorCollection;
  using cms::Ort::alpakatools::TensorSlice;

  // Run a model with an SOA_SCALAR input, presented as a [1] tensor, together with SoA columns, optionally in
  // mini-batches: the scalar has the same shape in every batch, including the last, partial one.
  class ScaledNet : public stream::FixedQueueEDProducer<> {
  public:
    ScaledNet(const edm::ParameterSet &params)
        : FixedQueueEDProducer<>(params),
          hits_token_(consumes(params.getParameter<edm::InputTag>("hits"))),
          scaled_net_token_{produces()},
          session_(params.getParameter<edm::FileInPath>("model").fullPath()),
          batch_size_(params.getParameter<uint32_t>("batchSize")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<uint32_t>("batchSize", 0)->setComment("Size of the mini-batches; 0 runs the inference on all the hits.");
      desc.add<edm::InputTag>("hits");
      descriptions.addWithDefaultLabel(desc);
    }

    void beginStream(edm::StreamID, Queue queue) override { session_.bind(queue); }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &hits = event.get(hits_token_);
      const auto total_size = hits.const_view().metadata().size();
      auto output = portabletest::SimpleNetDeviceCollection(event.queue(), total_size);
      const uint32_t n_batches = batch_size_ > 0 ? (total_size + batch_size_ - 1) / batch_size_ : 1;

      // records
      auto hit_records = hits.const_view().records();
      auto output_records = output.view().records();

      for (uint32_t batch_id = 0; batch_id < n_batches; ++batch_id) {
        // all the hits, or a mini-batch
        TensorSlice slice = batch_size_ > 0 ? TensorSlice{batch_id, batch_size_} : TensorSlice{};
        // input tensor definitions: [batch, 3] positions from three SOA_COLUMNs, and a [1] scale from an SOA_SCALAR
        TensorCollection<Queue> inputs;
        inputs.add<portabletest::HitSoA>("positions", slice, hit_records.x(), hit_records.y(), hit_records.z());
        inputs.add<portabletest::HitSoA>("scale", hit_records.hitScale());
        // output tensor definition: [batch, 1]
        TensorCollection<Queue> outputs;
        outputs.add<portabletest::SimpleNetSoA>("result", slice, output_records.reco_pt());

        session_.forward(event.queue(), inputs, outputs);
      }

      // put device-side product into event
      event.emplace(scaled_net_token_, std::move(output));
    }

  private:
    const device::EDGetToken<portabletest::HitDeviceCollection> hits_token_;
    const device::EDPutToken<portabletest::SimpleNetDeviceCollection> scaled_net_token_;
    ort::AlpakaSession session_;
    const uint32_t batch_size_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest

DEFINE_FWK_ALPAKA_MODULE(onnxtest::ScaledNet);
