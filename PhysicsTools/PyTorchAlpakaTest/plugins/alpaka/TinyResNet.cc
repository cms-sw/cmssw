#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Environment.h"
#include "PhysicsTools/PyTorchAlpaka/interface/NvtxRAII.h"
#include "PhysicsTools/PyTorchAlpaka/interface/QueueGuard.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace torchportabletest;
  using namespace cms::torch::alpakatools;
  using namespace cms::torchcommon;

  class TinyResNet : public stream::EDProducer<> {
  public:
    TinyResNet(const edm::ParameterSet &params)
        : EDProducer<>(params),
          images_token_(consumes(params.getParameter<edm::InputTag>("images"))),
          logits_token_{produces()},
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          environment_{static_cast<Environment>(params.getUntrackedParameter<int>("environment"))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<edm::InputTag>("images");
      desc.addUntracked<int>("environment", static_cast<int>(Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      NvtxRAII produce_range("TinyResNet::produce", environment_);
      NvtxRAII mem_alloc("TinyResNet::mem_alloc", environment_);

      // in/out collections
      const auto &images = event.get(images_token_);
      const auto batch_size = images.const_view().metadata().size();
      auto logits = LogitsDeviceCollection(batch_size, event.queue());
      mem_alloc.end();

      NvtxRAII metadata_def("TinyResNet::metadata_def", environment_);
      // records
      auto input_records = images.const_view().records();
      auto output_records = logits.view().records();
      // input tensor definition
      SoAMetadata inputs_metadata(batch_size);
      inputs_metadata.append_block<Image>("images", input_records.r(), input_records.g(), input_records.b());
      // output tensor definition
      SoAMetadata outputs_metadata(batch_size);
      outputs_metadata.append_block<Logits>("logits", output_records.logits());
      // metadata for automatic tensor conversion
      ModelMetadata metadata(inputs_metadata, outputs_metadata);
      metadata_def.end();

      // inference, queue guard restore stream when goes out of scope
      {
        QueueGuard<Queue> guard(event.queue());
        NvtxRAII mmove("TinyResNet::mmove", environment_);
        model_.to(event.queue());
        mmove.end();

        NvtxRAII inference("TinyResNet::inference", environment_);
        model_.forward(event.queue(), metadata);
        inference.end();
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
    const Environment environment_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::TinyResNet);