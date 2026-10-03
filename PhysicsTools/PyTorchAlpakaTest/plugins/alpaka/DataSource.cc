#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/HitDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/Environment.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/CommonKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  class DataSource : public stream::FixedQueueEDProducer<> {
  public:
    DataSource(const edm::ParameterSet &params)
        : FixedQueueEDProducer<>(params),
          particles_token_{produces()},
          hits_token_{produces()},
          hit_to_track_token_{produces()},
          images_token_{produces()},
          total_size_(params.getParameter<uint32_t>("totalSize")),
          hits_per_track_(params.getParameter<uint32_t>("hitsPerTrack")),
          environment_{static_cast<::torchtest::Environment>(params.getUntrackedParameter<int>("environment"))} {}

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // allocate data sources
      auto particles = portabletest::ParticleDeviceCollection(event.queue(), total_size_);
      const auto n_hits = total_size_ * hits_per_track_;
      auto hits = portabletest::HitDeviceCollection(event.queue(), n_hits);
      auto hit_to_track = portabletest::HitToTrackDeviceCollection(event.queue(), n_hits);
      auto images = portabletest::ImageDeviceCollection(event.queue(), total_size_);

      // fill data
      kernels::randomFillParticleCollection(event.queue(), particles);
      kernels::randomFillHitCollection(event.queue(), hits, hit_to_track, hits_per_track_);
      kernels::randomFillImageCollection(event.queue(), images);

      // put device-side data into event
      event.emplace(particles_token_, std::move(particles));
      event.emplace(hits_token_, std::move(hits));
      event.emplace(hit_to_track_token_, std::move(hit_to_track));
      event.emplace(images_token_, std::move(images));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<uint32_t>("totalSize");
      desc.add<uint32_t>("hitsPerTrack");
      desc.addUntracked<int>("environment", static_cast<int>(::torchtest::Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDPutToken<portabletest::ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<portabletest::HitDeviceCollection> hits_token_;
    const device::EDPutToken<portabletest::HitToTrackDeviceCollection> hit_to_track_token_;
    const device::EDPutToken<portabletest::ImageDeviceCollection> images_token_;
    const uint32_t total_size_;
    const uint32_t hits_per_track_;
    const ::torchtest::Environment environment_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::DataSource);
