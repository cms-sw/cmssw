#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/SimpleNetDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/MaskDeviceCollection.h"
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
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/CommonKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  class MaskedNet : public stream::EDProducer<> {
  public:
    MaskedNet(const edm::ParameterSet &params)
        : EDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          masked_net_token_{produces()},
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          environment_{static_cast<::torchtest::Environment>(params.getUntrackedParameter<int>("environment"))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<edm::InputTag>("particles");
      desc.addUntracked<int>("environment", static_cast<int>(::torchtest::Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &particles = event.get(particles_token_);
      const auto batch_size = particles.const_view().metadata().size();
      auto masked_net_output = portabletest::SimpleNetDeviceCollection(batch_size, event.queue());

      // mask
      auto mask = portabletest::MaskDeviceCollection(batch_size, event.queue());
      kernels::fillMask(event.queue(), mask);
      // note that scalar mask can be used to mask out entire batch at once (scalars are broadcasted)
      // auto scalar_mask = ScalarMaskDeviceCollection(batch_size, event.queue());
      // scalar_mask.zeroInitialise(event.queue());

      // records
      auto particle_records = particles.const_view().records();
      auto mask_records = mask.view().records();
      // auto scalar_mask_records = scalar_mask.view().records();
      auto output_records = masked_net_output.view().records();
      // input tensor definition
      cms::torch::alpakatools::TensorCollection<Queue> inputs(batch_size);
      inputs.add<portabletest::ParticleSoA>(
          "particles", particle_records.pt(), particle_records.eta(), particle_records.phi());
      // note override of default `ParticleSoA` layout with `MaskSoA`
      inputs.add<portabletest::MaskSoA>("mask", mask_records.mask());
      // inputs.add<ScalarMaskSoA>("scalar_mask", scalar_mask_records.scalar_mask());
      // output tensor definition
      cms::torch::alpakatools::TensorCollection<Queue> outputs(batch_size);
      outputs.add<portabletest::SimpleNetSoA>("regression_head", output_records.reco_pt());
      // metadata for automatic tensor conversion
      // ModelMetadata metadata(inputs, outputs);

      model_.forward(event.queue(), inputs, outputs);
      // put device-side product into event
      event.emplace(masked_net_token_, std::move(masked_net_output));
    }

  private:
    // event query tokens
    const device::EDGetToken<portabletest::ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<portabletest::SimpleNetDeviceCollection> masked_net_token_;
    // model
    torch::AlpakaModel model_;
    // debug mode flag
    const ::torchtest::Environment environment_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::MaskedNet);
