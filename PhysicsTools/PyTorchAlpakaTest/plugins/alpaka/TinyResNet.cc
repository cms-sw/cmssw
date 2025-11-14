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
#include "PhysicsTools/PyTorchAlpaka/interface/QueueGuard.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorRegistry.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/Environment.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace portabletest;
  using namespace cms::torch::alpakatools;

  class TinyResNet : public stream::EDProducer<> {
  public:
    TinyResNet(const edm::ParameterSet &params)
        : EDProducer<>(params),
          images_token_(consumes(params.getParameter<edm::InputTag>("images"))),
          logits_token_{produces()},
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          environment_{static_cast<::torchtest::Environment>(params.getUntrackedParameter<int>("environment"))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<edm::InputTag>("images");
      desc.addUntracked<int>("environment", static_cast<int>(::torchtest::Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &images = event.get(images_token_);
      const auto batch_size = images.const_view().metadata().size();
      auto logits = LogitsDeviceCollection(batch_size, event.queue());

      // records
      auto input_records = images.const_view().records();
      auto output_records = logits.view().records();
      // input tensor definition
      TensorRegistry<Queue> inputs(batch_size);
      inputs.register_tensor<ImageSoA>("images", input_records.r(), input_records.g(), input_records.b());
      // output tensor definition
      TensorRegistry<Queue> outputs(batch_size);
      outputs.register_tensor<LogitsSoA>("logits", output_records.logits());

      // inference, queue guard restore stream when goes out of scope
      {
        QueueGuard<Queue> guard(event.queue());
        model_.to(event.queue());
        model_.forward(event.queue(), inputs, outputs);
      }

      // put device-side product into event
      event.emplace(logits_token_, std::move(logits));
    }

  private:
    // event query tokens
    const device::EDGetToken<ImageDeviceCollection> images_token_;
    const device::EDPutToken<LogitsDeviceCollection> logits_token_;
    // model
    torch::AlpakaModel model_;
    // debug mode flag
    const ::torchtest::Environment environment_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::TinyResNet);
