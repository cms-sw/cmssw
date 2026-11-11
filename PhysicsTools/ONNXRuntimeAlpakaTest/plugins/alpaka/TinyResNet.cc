#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/LogitsDeviceCollection.h"
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

  class TinyResNet : public stream::FixedQueueEDProducer<> {
  public:
    TinyResNet(const edm::ParameterSet &params)
        : FixedQueueEDProducer<>(params),
          images_token_(consumes(params.getParameter<edm::InputTag>("images"))),
          logits_token_{produces()},
          session_(params.getParameter<edm::FileInPath>("model").fullPath()),
          batch_size_(params.getParameter<uint32_t>("batchSize")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<edm::InputTag>("images");
      desc.add<uint32_t>("batchSize", 0)
          ->setComment("Size of the mini-batches; 0 runs the inference on all the images.");
      descriptions.addWithDefaultLabel(desc);
    }

    void beginStream(edm::StreamID, Queue queue) override { session_.bind(queue); }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &images = event.get(images_token_);
      const auto total_size = images.const_view().metadata().size();
      auto logits = portabletest::LogitsDeviceCollection(event.queue(), total_size);
      const uint32_t n_batches = batch_size_ > 0 ? (total_size + batch_size_ - 1) / batch_size_ : 1;

      // records
      auto input_records = images.const_view().records();
      auto output_records = logits.view().records();

      for (uint32_t batch_id = 0; batch_id < n_batches; ++batch_id) {
        // all the images, or a mini-batch
        TensorSlice slice = batch_size_ > 0 ? TensorSlice{batch_id, batch_size_} : TensorSlice{};
        // input tensor definition: [batch, 3, 9, 9] images, from three Eigen 9x9 matrix columns
        TensorCollection<Queue> inputs;
        inputs.add<portabletest::ImageSoA>("images", slice, input_records.r(), input_records.g(), input_records.b());
        // output tensor definition: [batch, 10] logits, from an Eigen vector column
        TensorCollection<Queue> outputs;
        outputs.add<portabletest::LogitsSoA>("logits", slice, output_records.logits());

        session_.forward(event.queue(), inputs, outputs);
      }

      // put device-side product into event
      event.emplace(logits_token_, std::move(logits));
    }

  private:
    const device::EDGetToken<portabletest::ImageDeviceCollection> images_token_;
    const device::EDPutToken<portabletest::LogitsDeviceCollection> logits_token_;
    ort::AlpakaSession session_;
    const uint32_t batch_size_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest

DEFINE_FWK_ALPAKA_MODULE(onnxtest::TinyResNet);
