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

  class MultiHeadNet : public stream::EDProducer<> {
  public:
    MultiHeadNet(const edm::ParameterSet &params)
        : EDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          multi_head_net_token_{produces()},
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          environment_{static_cast<Environment>(params.getUntrackedParameter<int>("environment"))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<edm::InputTag>("particles");
      desc.addUntracked<int>("environment", static_cast<int>(Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      NvtxRAII produce_range("MultiHeadNet::produce", environment_);
      NvtxRAII mem_alloc("MultiHeadNet::mem_alloc", environment_);

      // in/out collections
      const auto &particles = event.get(particles_token_);
      const auto batch_size = particles.const_view().metadata().size();
      auto multi_head_output = MultiHeadNetDeviceCollection(batch_size, event.queue());
      mem_alloc.end();

      NvtxRAII metadata_def("MultiHeadNet::metadata_def", environment_);
      // records
      auto input_records = particles.const_view().records();
      auto output_records = multi_head_output.view().records();
      // input tensor definition
      SoAMetadata inputs_metadata(batch_size);
      inputs_metadata.append_block<ParticleSoA>(
          "particles", input_records.pt(), input_records.eta(), input_records.phi());
      // output tensor definition
      SoAMetadata outputs_metadata(batch_size);
      outputs_metadata.append_block<MultiHeadNetSoA>("regression_head", output_records.regression_head());
      outputs_metadata.append_block<MultiHeadNetSoA>("classification_head", output_records.classification_head());
      // metadata for automatic tensor conversion
      // note that `multi_head` is true to distinguish the multi-branch backward conversion
      ModelMetadata metadata(inputs_metadata, outputs_metadata, /**< multi_head = */ true);
      metadata_def.end();

      // inference, queue guard restore stream when goes out of scope
      {
        QueueGuard<Queue> guard(event.queue());
        NvtxRAII mmove("MultiHeadNet::mmove", environment_);
        model_.to(event.queue());
        mmove.end();

        NvtxRAII inference("MultiHeadNet::inference", environment_);
        model_.forward(event.queue(), metadata);
        inference.end();
      }

      // put device-side product into event
      event.emplace(multi_head_net_token_, std::move(multi_head_output));
    }

  private:
    // event query tokens
    const device::EDGetToken<ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<MultiHeadNetDeviceCollection> multi_head_net_token_;
    // model
    torch::AlpakaModel model_;
    // debug mode flag
    const Environment environment_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::MultiHeadNet);