#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/MultiHeadNetDeviceCollection.h"
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

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  class MultiHeadNet : public stream::EDProducer<> {
  public:
    MultiHeadNet(const edm::ParameterSet &params)
        : EDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          multi_head_net_token_{produces()},
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
      auto multi_head_output = portabletest::MultiHeadNetDeviceCollection(batch_size, event.queue());

      // records
      auto input_records = particles.const_view().records();
      auto output_records = multi_head_output.view().records();
      // input tensor definition
      cms::torch::alpakatools::TensorCollection<Queue> inputs(batch_size);
      inputs.add<portabletest::ParticleSoA>("particles", input_records.pt(), input_records.eta(), input_records.phi());
      // output tensor definition
      cms::torch::alpakatools::TensorCollection<Queue> outputs(batch_size);
      outputs.add<portabletest::MultiHeadNetSoA>("regression_head", output_records.regression_head());
      outputs.add<portabletest::MultiHeadNetSoA>("classification_head", output_records.classification_head());

      model_.forward(event.queue(), inputs, outputs);
      // put device-side product into event
      event.emplace(multi_head_net_token_, std::move(multi_head_output));
    }

  private:
    // event query tokens
    const device::EDGetToken<portabletest::ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<portabletest::MultiHeadNetDeviceCollection> multi_head_net_token_;
    // model
    torch::AlpakaModel model_;
    // debug mode flag
    const ::torchtest::Environment environment_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::MultiHeadNet);
