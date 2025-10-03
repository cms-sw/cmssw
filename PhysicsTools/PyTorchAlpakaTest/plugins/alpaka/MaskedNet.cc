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
#include "PhysicsTools/PyTorchAlpakaTest/interface/alpaka/MaskDevice.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/CommonKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace torchportabletest;
  using namespace cms::torch::alpakatools;
  using namespace cms::torchcommon;

  class MaskedNet : public stream::EDProducer<> {
  public:
    MaskedNet(const edm::ParameterSet &params)
        : EDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          masked_net_token_{produces()},
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
      NvtxRAII produce_range("MaskedNet::produce", environment_);
      NvtxRAII mem_alloc("MaskedNet::mem_alloc", environment_);

      // in/out collections
      const auto &particles = event.get(particles_token_);
      const auto batch_size = particles.const_view().metadata().size();
      auto masked_net_output = SimpleNetDeviceCollection(batch_size, event.queue());

      // mask
      auto mask = MaskDevice(batch_size, event.queue());
      kernels::fillMask(event.queue(), mask);
      // note that scalar mask can be used to mask out entire batch at once (scalars are broadcasted)
      // auto scalar_mask = ScalarMaskDevice(batch_size, event.queue());
      // scalar_mask.zeroInitialise(event.queue());
      mem_alloc.end();

      NvtxRAII metadata_def("MaskedNet::metadata_def", environment_);
      // records
      auto particle_records = particles.const_view().records();
      auto mask_records = mask.view().records();
      // auto scalar_mask_records = scalar_mask.view().records();
      auto output_records = masked_net_output.view().records();
      // input tensor definition
      SoAMetadata inputs_metadata(batch_size);
      inputs_metadata.append_block<ParticleSoA>(
          "particles", batch_size, particle_records.pt(), particle_records.eta(), particle_records.phi());
      // note override of default `ParticleSoA` layout with `Mask`
      inputs_metadata.append_block<Mask>("mask", mask_records.mask());
      // inputs_metadata.append_block<ScalarMask>("scalar_mask", scalar_mask_records.scalar_mask());
      // output tensor definition
      SoAMetadata outputs_metadata(batch_size);
      outputs_metadata.append_block<SimpleNetSoA>("regression_head", output_records.reco_pt());
      // metadata for automatic tensor conversion
      ModelMetadata metadata(inputs_metadata, outputs_metadata);
      metadata_def.end();

      // inference, queue guard restore stream when goes out of scope
      {
        QueueGuard<Queue> guard(event.queue());
        NvtxRAII mmove("MaskedNet::mmove", environment_);
        model_.to(event.queue());
        mmove.end();

        NvtxRAII inference("MaskedNet::inference", environment_);
        model_.forward(event.queue(), metadata);
        inference.end();
      }

      // put device-side product into event
      event.emplace(masked_net_token_, std::move(masked_net_output));
    }

  private:
    // event query tokens
    const device::EDGetToken<ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<SimpleNetDeviceCollection> masked_net_token_;
    // model
    torch::AlpakaModel model_;
    // debug mode flag
    const Environment environment_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::MaskedNet);