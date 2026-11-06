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

  class TinyResNet : public stream::FixedQueueEDProducer<> {
  public:
    TinyResNet(const edm::ParameterSet &params)
        : FixedQueueEDProducer<>(params),
          images_token_(consumes(params.getParameter<edm::InputTag>("images"))),
          logits_token_{produces()},
          session_(params.getParameter<edm::FileInPath>("model").fullPath()),
          batch_size_(params.getParameter<int>("batchSize")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<edm::InputTag>("images");
      desc.add<int>("batchSize", 0)->setComment("Size of the mini-batches; 0 runs the inference on all the images.");
      descriptions.addWithDefaultLabel(desc);
    }

    void beginStream(edm::StreamID, Queue queue) override { session_.bind(queue); }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &images = event.get(images_token_);
      const auto total_size = images.const_view().metadata().size();
      auto logits = portabletest::LogitsDeviceCollection(event.queue(), total_size);
      const int batch_size = batch_size_ > 0 ? batch_size_ : total_size;
      const int n_batches = batch_size > 0 ? (total_size + batch_size - 1) / batch_size : 0;

      // records
      auto input_records = images.const_view().records();
      auto output_records = logits.view().records();

      for (int i_batch = 0; i_batch < n_batches; ++i_batch) {
        // input tensor definition: [batch, 3, 9, 9] images, from three Eigen 9x9 matrix columns
        TensorCollection<Queue> inputs(batch_size, total_size);
        inputs.add<portabletest::ImageSoA>("images", i_batch, input_records.r(), input_records.g(), input_records.b());
        // output tensor definition: [batch, 10] logits, from an Eigen vector column
        TensorCollection<Queue> outputs(batch_size, total_size);
        outputs.add<portabletest::LogitsSoA>("logits", i_batch, output_records.logits());

        session_.forward(event.queue(), inputs, outputs);
      }

      // put device-side product into event
      event.emplace(logits_token_, std::move(logits));
    }

  private:
    const device::EDGetToken<portabletest::ImageDeviceCollection> images_token_;
    const device::EDPutToken<portabletest::LogitsDeviceCollection> logits_token_;
    ort::AlpakaSession session_;
    const int batch_size_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest

DEFINE_FWK_ALPAKA_MODULE(onnxtest::TinyResNet);
