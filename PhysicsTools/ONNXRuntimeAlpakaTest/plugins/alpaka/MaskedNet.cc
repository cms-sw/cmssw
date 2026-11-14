#include "DataFormats/PortableTestObjects/interface/alpaka/MaskDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
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
#include "PhysicsTools/ONNXRuntimeAlpakaTest/plugins/alpaka/CommonKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest {

  using cms::Ort::alpakatools::TensorCollection;

  class MaskedNet : public stream::FixedQueueEDProducer<> {
  public:
    MaskedNet(const edm::ParameterSet &params)
        : FixedQueueEDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          masked_net_token_{produces()},
          session_(params.getParameter<edm::FileInPath>("model").fullPath()),
          reverse_order_(params.getParameter<bool>("reverseOrder")),
          empty_mask_(params.getParameter<bool>("emptyMask")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<edm::InputTag>("particles");
      desc.add<bool>("reverseOrder", false)
          ->setComment("Add the input tensors in the opposite order of the model inputs: they are matched by name.");
      desc.add<bool>("emptyMask", false)
          ->setComment("Use an empty mask: the inference is skipped, and the output is filled with zeros.");
      descriptions.addWithDefaultLabel(desc);
    }

    void beginStream(edm::StreamID, Queue queue) override { session_.bind(queue); }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &particles = event.get(particles_token_);
      const auto total_size = particles.const_view().metadata().size();
      auto masked_net_output = portabletest::SimpleNetDeviceCollection(event.queue(), total_size);

      // mask
      auto mask = portabletest::MaskDeviceCollection(event.queue(), empty_mask_ ? 0 : total_size);
      kernels::fillMask(event.queue(), mask);
      if (empty_mask_) {
        // fill the output with NaNs, to check that it is filled with zeros
        auto buffer = masked_net_output.buffer();
        alpaka::memset(event.queue(), buffer, 0xff);
      }

      // records
      auto particle_records = particles.const_view().records();
      auto mask_records = mask.view().records();
      auto output_records = masked_net_output.view().records();
      // input tensor definitions: float particles and uint8_t mask
      TensorCollection<Queue> inputs;
      if (reverse_order_) {
        inputs.add<portabletest::MaskSoA>("mask", mask_records.mask());
      }
      inputs.add<portabletest::ParticleSoA>(
          "particles", particle_records.pt(), particle_records.eta(), particle_records.phi());
      if (not reverse_order_) {
        inputs.add<portabletest::MaskSoA>("mask", mask_records.mask());
      }
      // output tensor definition
      TensorCollection<Queue> outputs;
      outputs.add<portabletest::SimpleNetSoA>("regression_head", output_records.reco_pt());

      session_.forward(event.queue(), inputs, outputs);
      // put device-side product into event
      event.emplace(masked_net_token_, std::move(masked_net_output));
    }

  private:
    const device::EDGetToken<portabletest::ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<portabletest::SimpleNetDeviceCollection> masked_net_token_;
    ort::AlpakaSession session_;
    const bool reverse_order_;
    const bool empty_mask_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest

DEFINE_FWK_ALPAKA_MODULE(onnxtest::MaskedNet);
