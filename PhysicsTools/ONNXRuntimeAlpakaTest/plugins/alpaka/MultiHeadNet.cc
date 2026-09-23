#include "DataFormats/PortableTestObjects/interface/alpaka/MultiHeadNetDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
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

  class MultiHeadNet : public stream::FixedQueueEDProducer<> {
  public:
    MultiHeadNet(const edm::ParameterSet &params)
        : FixedQueueEDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          multi_head_net_token_{produces()},
          session_(params.getParameter<edm::FileInPath>("model").fullPath()) {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<edm::InputTag>("particles");
      descriptions.addWithDefaultLabel(desc);
    }

    void beginStream(edm::StreamID, Queue queue) override { session_.bind(queue); }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &particles = event.get(particles_token_);
      const auto total_size = particles.const_view().metadata().size();
      auto multi_head_output = portabletest::MultiHeadNetDeviceCollection(event.queue(), total_size);

      // records
      auto input_records = particles.const_view().records();
      auto output_records = multi_head_output.view().records();
      // input tensor definition
      TensorCollection<Queue> inputs(total_size);
      inputs.add<portabletest::ParticleSoA>("particles", input_records.pt(), input_records.eta(), input_records.phi());
      // output tensor definitions: both outputs are written directly into the SoA
      TensorCollection<Queue> outputs(total_size);
      outputs.add<portabletest::MultiHeadNetSoA>("regression_head", output_records.regression_head());
      outputs.add<portabletest::MultiHeadNetSoA>("classification_head", output_records.classification_head());

      session_.forward(event.queue(), inputs, outputs);
      // put device-side product into event
      event.emplace(multi_head_net_token_, std::move(multi_head_output));
    }

  private:
    const device::EDGetToken<portabletest::ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<portabletest::MultiHeadNetDeviceCollection> multi_head_net_token_;
    ort::AlpakaSession session_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest

DEFINE_FWK_ALPAKA_MODULE(onnxtest::MultiHeadNet);
