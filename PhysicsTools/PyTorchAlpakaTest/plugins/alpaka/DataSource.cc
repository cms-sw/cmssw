#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/Environment.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/CommonKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  class DataSource : public stream::EDProducer<> {
  public:
    DataSource(const edm::ParameterSet &params)
        : EDProducer<>(params),
          particles_token_{produces()},
          images_token_{produces()},
          batch_size_(params.getParameter<uint32_t>("batchSize")),
          environment_{static_cast<::torchtest::Environment>(params.getUntrackedParameter<int>("environment"))} {}

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // allocate data sources
      auto particles = portabletest::ParticleDeviceCollection(batch_size_, event.queue());
      auto images = portabletest::ImageDeviceCollection(batch_size_, event.queue());

      // fill data
      kernels::randomFillParticleCollection(event.queue(), particles);
      kernels::randomFillImageCollection(event.queue(), images);

      // put device-side data into event
      event.emplace(particles_token_, std::move(particles));
      event.emplace(images_token_, std::move(images));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<uint32_t>("batchSize");
      desc.addUntracked<int>("environment", static_cast<int>(::torchtest::Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDPutToken<portabletest::ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<portabletest::ImageDeviceCollection> images_token_;
    const uint32_t batch_size_;
    const ::torchtest::Environment environment_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::DataSource);
