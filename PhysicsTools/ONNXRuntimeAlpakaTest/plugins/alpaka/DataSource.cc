#include <vector>

#include "DataFormats/PortableTestObjects/interface/alpaka/HitDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
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
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/ONNXRuntimeAlpakaTest/plugins/alpaka/CommonKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest {

  // Produce the particles, images and hits used by the tests, with random values that change from event to event.
  // The number of elements of event i is sizes[(i - 1) % sizes.size()], so the size can also change from event to
  // event.

  class DataSource : public stream::FixedQueueEDProducer<> {
  public:
    DataSource(const edm::ParameterSet &params)
        : FixedQueueEDProducer<>(params),
          particles_token_{produces()},
          images_token_{produces()},
          hits_token_{produces()},
          sizes_(params.getParameter<std::vector<uint32_t>>("sizes")) {
      if (sizes_.empty())
        throw cms::Exception("Configuration") << "The list of sizes must not be empty";
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      const auto id = event.id().event();
      const uint32_t size = sizes_[(id - 1) % sizes_.size()];
      const uint32_t seed = static_cast<uint32_t>(id * 3);

      // allocate data sources
      auto particles = portabletest::ParticleDeviceCollection(event.queue(), size);
      auto images = portabletest::ImageDeviceCollection(event.queue(), size);
      auto hits = portabletest::HitDeviceCollection(event.queue(), size);

      // fill data
      kernels::randomFillParticleCollection(event.queue(), particles, seed);
      kernels::randomFillImageCollection(event.queue(), images, seed + 1);
      kernels::randomFillHitCollection(event.queue(), hits, seed + 2);

      // put device-side data into event
      event.emplace(particles_token_, std::move(particles));
      event.emplace(images_token_, std::move(images));
      event.emplace(hits_token_, std::move(hits));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::vector<uint32_t>>("sizes", {35})->setComment("Number of elements in each event, used cyclically.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDPutToken<portabletest::ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<portabletest::ImageDeviceCollection> images_token_;
    const device::EDPutToken<portabletest::HitDeviceCollection> hits_token_;
    const std::vector<uint32_t> sizes_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest

DEFINE_FWK_ALPAKA_MODULE(onnxtest::DataSource);
