#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/LogitsDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/Environment.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {
  struct BatchIO {
    cms::torch::alpakatools::TensorCollection<Queue> inputs;
    cms::torch::alpakatools::TensorCollection<Queue> outputs;
  };

  class TinyResNetMiniBatch : public stream::EDProducer<> {
  public:
    TinyResNetMiniBatch(const edm::ParameterSet &params)
        : EDProducer<>(params),
          images_token_(consumes(params.getParameter<edm::InputTag>("images"))),
          logits_token_{produces()},
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          total_size_(params.getParameter<int>("batchSize")),
          batch_size_(params.getParameter<int>("miniBatchSize")),
          environment_{static_cast<::torchtest::Environment>(params.getUntrackedParameter<int>("environment"))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<int>("batchSize");
      desc.add<int>("miniBatchSize");
      desc.add<edm::InputTag>("images");
      desc.addUntracked<int>("environment", static_cast<int>(::torchtest::Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &images = event.get(images_token_);
      auto logits = portabletest::LogitsDeviceCollection(event.queue(), total_size_);

      auto n_batches = total_size_ / batch_size_;
      // records
      auto input_records = images.const_view().records();
      auto output_records = logits.view().records();

      // input and output tensor definitions
      std::deque<BatchIO> batches;
      for (int i_batch = 0; i_batch < n_batches; ++i_batch) {
        BatchIO batch{cms::torch::alpakatools::TensorCollection<Queue>(batch_size_, total_size_),
                      cms::torch::alpakatools::TensorCollection<Queue>(batch_size_, total_size_)};

        batch.inputs.add<portabletest::ImageSoA>(
            "images", i_batch, input_records.r(), input_records.g(), input_records.b());

        batch.outputs.add<portabletest::LogitsSoA>("logits", i_batch, output_records.logits());
        batches.push_back(std::move(batch));
      }

      // forward pass on mini-batches
      for (auto &batch : batches) {
        model_.forward(event.queue(), batch.inputs, batch.outputs);
      }

      // put device-side product into event
      event.emplace(logits_token_, std::move(logits));
    }

  private:
    // event query tokens
    const device::EDGetToken<portabletest::ImageDeviceCollection> images_token_;
    const device::EDPutToken<portabletest::LogitsDeviceCollection> logits_token_;
    // model
    torch::AlpakaModel model_;
    const int total_size_;
    const int batch_size_;
    // debug mode flag
    const ::torchtest::Environment environment_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::TinyResNetMiniBatch);
