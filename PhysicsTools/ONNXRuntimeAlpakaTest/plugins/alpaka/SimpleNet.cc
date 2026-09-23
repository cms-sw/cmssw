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

namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest {

  using cms::Ort::alpakatools::Layout;
  using cms::Ort::alpakatools::TensorCollection;

  class SimpleNet : public stream::FixedQueueEDProducer<> {
  public:
    SimpleNet(const edm::ParameterSet &params)
        : FixedQueueEDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          simple_net_token_{produces()},
          session_(params.getParameter<edm::FileInPath>("model").fullPath()),
          feature_major_(params.getParameter<bool>("featureMajor")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<edm::InputTag>("particles");
      desc.add<bool>("featureMajor", false)
          ->setComment("Pass the SoA memory directly to a model exported to accept and return transposed tensors.");
      descriptions.addWithDefaultLabel(desc);
    }

    // bind the ONNX Runtime session to the queue used by this module for all events
    void beginStream(edm::StreamID, Queue queue) override { session_.bind(queue); }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &particles = event.get(particles_token_);
      const auto total_size = particles.const_view().metadata().size();
      auto regression_collection = portabletest::SimpleNetDeviceCollection(event.queue(), total_size);

      // records
      auto input_records = particles.const_view().records();
      auto output_records = regression_collection.view().records();
      // input tensor definition
      TensorCollection<Queue> inputs(total_size);
      inputs.add<portabletest::ParticleSoA>("particles", input_records.pt(), input_records.eta(), input_records.phi());
      // output tensor definition
      TensorCollection<Queue> outputs(total_size);
      outputs.add<portabletest::SimpleNetSoA>("regression_head", output_records.reco_pt());
      if (feature_major_) {
        inputs.set_layout("particles", Layout::FeatureMajor);
        outputs.set_layout("regression_head", Layout::FeatureMajor);
      }

      session_.forward(event.queue(), inputs, outputs);
      // put device-side product into event
      event.emplace(simple_net_token_, std::move(regression_collection));
    }

  private:
    const device::EDGetToken<portabletest::ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<portabletest::SimpleNetDeviceCollection> simple_net_token_;
    ort::AlpakaSession session_;
    const bool feature_major_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest

DEFINE_FWK_ALPAKA_MODULE(onnxtest::SimpleNet);
