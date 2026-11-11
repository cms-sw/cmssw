#include <deque>

#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/SimpleNetDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/alpaka/AlpakaSession.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest {

  using cms::Ort::alpakatools::TensorCollection;
  using cms::Ort::alpakatools::TensorSlice;

  // This module uses a different queue for each event: the ONNX Runtime session is bound to the queue of the first
  // event, and synchronised with the queues of the following events using alpaka events.
  class SimpleNetMiniBatch : public stream::EDProducer<> {
  public:
    SimpleNetMiniBatch(const edm::ParameterSet &params)
        : EDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          simple_net_token_{produces()},
          session_(params.getParameter<edm::FileInPath>("model").fullPath()),
          batch_size_(params.getParameter<uint32_t>("batchSize")) {
      if (batch_size_ == 0)
        throw cms::Exception("Configuration") << "batchSize must be greater than zero";
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<uint32_t>("batchSize");
      desc.add<edm::InputTag>("particles");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // in/out collections
      const auto &particles = event.get(particles_token_);
      const auto total_size = particles.const_view().metadata().size();
      auto regression_collection = portabletest::SimpleNetDeviceCollection(event.queue(), total_size);
      const uint32_t n_batches = (total_size + batch_size_ - 1) / batch_size_;

      // records
      auto input_records = particles.const_view().records();
      auto output_records = regression_collection.view().records();

      // forward pass on mini-batches
      for (uint32_t batch_id = 0; batch_id < n_batches; ++batch_id) {
        TensorSlice slice{batch_id, batch_size_};
        TensorCollection<Queue> inputs;
        inputs.add<portabletest::ParticleSoA>(
            "particles", slice, input_records.pt(), input_records.eta(), input_records.phi());
        TensorCollection<Queue> outputs;
        outputs.add<portabletest::SimpleNetSoA>("regression_head", slice, output_records.reco_pt());
        session_.forward(event.queue(), inputs, outputs);
      }
      // put device-side product into event
      event.emplace(simple_net_token_, std::move(regression_collection));
    }

  private:
    const device::EDGetToken<portabletest::ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<portabletest::SimpleNetDeviceCollection> simple_net_token_;
    ort::AlpakaSession session_;
    const uint32_t batch_size_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest

DEFINE_FWK_ALPAKA_MODULE(onnxtest::SimpleNetMiniBatch);
