#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco {

  using HitPortableCollectionHost = PortableHostCollection<reco::TrackingBlocksSoA>;

  class TrackingRecHitHost : public HitPortableCollectionHost {
  public:
    explicit TrackingRecHitHost(edm::Uninitialized)
        : PortableHostCollection<reco::TrackingBlocksSoA>{edm::kUninitialized} {}

    // Constructor which specifies only the SoA size, to be used when copying
    // the results from the device to the host.
    // Construct the object in pageable system memory.
    explicit TrackingRecHitHost(alpaka_common::DevHost const& host, uint32_t nHits, uint32_t nModules)
        : HitPortableCollectionHost(host, nHits, nModules + 1) {}
    // Why this +1? See TrackingRecHitDevice.h constructor for an explanation

    // Constructor which specifies only the SoA size, to be used when copying
    // the results from the device to the host.
    // Construct the object in pinned host memory associated to the given work
    // queue, accessible by the queue's device.
    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    explicit TrackingRecHitHost(TQueue queue, uint32_t nHits, uint32_t nModules)
        : HitPortableCollectionHost(queue, nHits, nModules + 1) {}
    // Why this +1? See TrackingRecHitDevice.h constructor for an explanation

    // Constructor from clusters
    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    explicit TrackingRecHitHost(TQueue queue, SiPixelClustersHost const& clusters)
        : HitPortableCollectionHost(queue, clusters.nClusters(), clusters.view().metadata().size()) {
      auto hitsView = view().trackingHits();
      auto modsView = view().hitModules();

      auto nModules = clusters.view().metadata().size();

      auto clusters_m = cms::alpakatools::make_host_view(clusters.view().clusModuleStart(), nModules);
      auto hits_m = cms::alpakatools::make_host_view(modsView.moduleStart(), nModules);

      alpaka::memcpy(queue, hits_m, clusters_m);

      hitsView.offsetBPIX2() = clusters.offsetBPIX2();
    }

    uint32_t nHits() const { return this->view().trackingHits().metadata().size(); }
    uint32_t nModules() const { return this->view().hitModules().metadata().size() - 1; }

    int32_t offsetBPIX2() const { return this->view().trackingHits().offsetBPIX2(); }

    // do nothing for a host collection
    template <typename TQueue>
    void updateFromDevice(TQueue) {}
  };

}  // namespace reco

namespace ngt {

  template <>
  struct MemoryCopyTraits<reco::TrackingRecHitHost> {
    using value_type = reco::TrackingRecHitHost;

    struct Properties {
      uint32_t nHits;
      uint32_t nModules;
    };

    static Properties properties(value_type const& object) { return {object.nHits(), object.nModules()}; }

    static void initialize(value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pageable host memory.
      object = value_type(cms::alpakatools::host(), prop.nHits, prop.nModules);
    }

    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    static void initialize(TQueue& queue, value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pinned host memory.
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

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h
