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
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/Environment.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  class TinyResNetMiniBatch : public stream::EDProducer<> {
  public:
    TinyResNetMiniBatch(const edm::ParameterSet &params)
        : EDProducer<>(params),
          images_token_(consumes(params.getParameter<edm::InputTag>("images"))),
          logits_token_{produces()},
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          batch_size_(params.getParameter<uint32_t>("batchSize")),
          environment_{static_cast<::torchtest::Environment>(params.getUntrackedParameter<int>("environment"))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<uint32_t>("batchSize");
      desc.add<edm::InputTag>("images");
      desc.addUntracked<int>("environment", static_cast<int>(::torchtest::Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &images = event.get(images_token_);
      const auto total_size = images.const_view().metadata().size();
      auto logits = portabletest::LogitsDeviceCollection(event.queue(), total_size);

      // records
      auto input_records = images.const_view().records();
      auto output_records = logits.view().records();

      auto inputs = cms::torch::alpakatools::BatchedTensorCollection<Queue>();
      auto outputs = cms::torch::alpakatools::BatchedTensorCollection<Queue>();

      inputs.addBatched<portabletest::ImageSoA>(
          "images", batch_size_, input_records.r(), input_records.g(), input_records.b());
      outputs.addBatched<portabletest::LogitsSoA>("logits", batch_size_, output_records.logits());

      model_.forward(event.queue(), inputs, outputs);

      // put device-side product into event
      event.emplace(logits_token_, std::move(logits));
    }

  private:
    // event query tokens
    const device::EDGetToken<portabletest::ImageDeviceCollection> images_token_;
    const device::EDPutToken<portabletest::LogitsDeviceCollection> logits_token_;
    // model
    torch::AlpakaModel model_;
    const uint32_t batch_size_;
    // debug mode flag
    const ::torchtest::Environment environment_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::TinyResNetMiniBatch);
