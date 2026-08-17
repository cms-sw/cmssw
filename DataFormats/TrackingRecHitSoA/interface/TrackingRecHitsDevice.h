#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitSoADevice_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitSoADevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco {

  template <typename TDev>
  using HitPortableCollectionDevice = PortableDeviceCollection<TDev, reco::TrackingBlocksSoA>;

  template <typename TDev>
  class TrackingRecHitDevice : public HitPortableCollectionDevice<TDev> {
  public:
    TrackingRecHitDevice() = default;

    TrackingRecHitDevice(edm::Uninitialized) : HitPortableCollectionDevice<TDev>{edm::kUninitialized} {}

    // Constructor which specifies only the SoA size, to be used when copying the results from host to device
    template <typename TQueue>
    explicit TrackingRecHitDevice(TQueue queue, uint32_t nHits, uint32_t nModules)
        : HitPortableCollectionDevice<TDev>(queue, nHits, nModules + 1) {}

    // N.B. why this + 1? Because the HitModulesLayout is holding the
    // moduleStart vector that is a cumulative sum of all the hits
    // in each module. The extra element of the array (the last one)
    // is used to hold the total number of hits. We are "hiding" this
    // in the constructor so that one can build the TrackingRecHit class
    // in a more natural way, just using the number of needed modules.

    // Constructor from clusters
    template <typename TQueue>
    explicit TrackingRecHitDevice(TQueue queue, SiPixelClustersDevice<TDev> const& clusters)
        : HitPortableCollectionDevice<TDev>(queue, clusters.nClusters(), clusters.view().metadata().size()),
          offsetBPIX2_{clusters.offsetBPIX2()} {
      auto hitsView = this->view().trackingHits();
      auto modsView = this->view().hitModules();

      auto nModules = clusters.view().metadata().size();

      auto clusters_m = cms::alpakatools::make_device_view(queue, clusters.view().clusModuleStart(), nModules);
      auto hits_m = cms::alpakatools::make_device_view(queue, modsView.moduleStart(), nModules);

      alpaka::memcpy(queue, hits_m, clusters_m);

      auto off_h = cms::alpakatools::make_host_view(offsetBPIX2_);
      auto off_d = cms::alpakatools::make_device_view(queue, hitsView.offsetBPIX2());
      alpaka::memcpy(queue, off_d, off_h);
    }

    uint32_t nHits() const { return static_cast<uint32_t>(this->view().trackingHits().metadata().size()); }
    uint32_t nModules() const { return static_cast<uint32_t>(this->view().hitModules().metadata().size() - 1); }

    int32_t offsetBPIX2() const { return offsetBPIX2_; }
    uint32_t offsetStubs() const { return offsetStubs_; }
    // Set the two cached offsets from values known on the host (no transfer); use updateFromDevice()
    // when the SoA scalars were written by a kernel.
    void setOffsets(int32_t offsetBPIX2, uint32_t offsetStubs) {
      offsetBPIX2_ = offsetBPIX2;
      offsetStubs_ = offsetStubs;
    }

    // Update the two cached offsets from the SoA scalars on the device. Blocking: the copies are
    // staged through pinned host buffers (a direct copy into the pageable members would synchronise
    // per copy) and followed by a single wait before the members are assigned.
    template <typename TQueue>
    void updateFromDevice(TQueue queue) {
      auto offBPIX2_h = cms::alpakatools::make_host_buffer<int32_t>(queue);
      auto offStubs_h = cms::alpakatools::make_host_buffer<uint32_t>(queue);

      auto offBPIX2_d = cms::alpakatools::make_device_view(queue, this->view().trackingHits().offsetBPIX2());
      alpaka::memcpy(queue, offBPIX2_h, offBPIX2_d);

      auto offStubs_d = cms::alpakatools::make_device_view(queue, this->view().trackingHits().offsetStubs());
      alpaka::memcpy(queue, offStubs_h, offStubs_d);

      alpaka::wait(queue);
      offsetBPIX2_ = *offBPIX2_h.data();
      offsetStubs_ = *offStubs_h.data();
    }

  private:
    // offsetBPIX2 is used on host functions so is useful to have it also stored in the class and not only in the layout
    int32_t offsetBPIX2_ = 0;
    // offsetStubs marks the start of stub-derived hits in the merged collection
    uint32_t offsetStubs_ = 0;
  };
}  // namespace reco

namespace ngt {

  template <typename TDev>
  struct MemoryCopyTraits<reco::TrackingRecHitDevice<TDev>> {
    using value_type = reco::TrackingRecHitDevice<TDev>;

    struct Properties {
      uint32_t nHits;
      uint32_t nModules;
    };

    static Properties properties(value_type const& object) { return {object.nHits(), object.nModules()}; }

    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    static void initialize(TQueue& queue, value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in device global memory.
      object = value_type(queue, prop.nHits, prop.nModules);
    }

    static std::vector<std::span<std::byte>> regions(value_type& object) {
      std::byte* address = reinterpret_cast<std::byte*>(object.buffer().data());
      size_t size = alpaka::getExtentProduct(object.buffer());
      return {{address, size}};
    }

    static std::vector<std::span<const std::byte>> regions(value_type const& object) {
      const std::byte* address = reinterpret_cast<const std::byte*>(object.buffer().data());
      size_t size = alpaka::getExtentProduct(object.buffer());
      return {{address, size}};
    }
  };

}  // namespace ngt

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitSoADevice_h
