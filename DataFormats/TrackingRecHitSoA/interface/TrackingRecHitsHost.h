#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco
{

  using HitPortableCollectionHost = PortableHostMultiCollection<reco::TrackingRecHitSoA, reco::HitModuleSoA>;

  class TrackingRecHitHost : public HitPortableCollectionHost {
  public:

    TrackingRecHitHost() = default;

    // Constructor which specifies only the SoA size, to be used when copying the results from the device to the host
    template <typename TQueue>
    explicit TrackingRecHitHost(TQueue queue, uint32_t nHits, uint32_t nModules)
        : HitPortableCollectionHost({{int(nHits),int(nModules)}}, queue) {}

    // Constructor from clusters
    template <typename TQueue>
    explicit TrackingRecHitHost(TQueue queue, SiPixelClustersHost const &clusters)
        : PortableHostMultiCollection({{int(clusters.nClusters()),clusters.view().metadata().size()}}, queue)  {
          
      auto hitsView = this->template view<TrackingRecHitSoA>();
      auto modsView = this->template view<HitModuleSoA>();

      auto nModules = clusters.view().metadata().size();
      
      auto clusters_m = cms::alpakatools::make_host_view(clusters.view().clusModuleStart(), nModules);
      auto hits_m = cms::alpakatools::make_host_view(modsView.moduleStart(), nModules);

      alpaka::memcpy(queue, hits_m, clusters_m);

      hitsView.offsetBPIX2() = clusters.offsetBPIX2();
    }

    uint32_t nHits() const { return this->template view<TrackingRecHitSoA>().metadata().size(); }
    uint32_t nModules() const { return this->template view<HitModuleSoA>().metadata().size(); }

    int32_t offsetBPIX2() const { return this->template view<TrackingRecHitSoA>().offsetBPIX2(); }

    uint32_t const* hitsModuleStart() const { return this->template view<TrackingRecHitSoA>().hitsModuleStart().data(); }

    // do nothing for a host collection
    template <typename TQueue>
    void updateFromDevice(TQueue) {}
  };

 

}

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h
