// C++ headers
#include <cassert>
#include <cstdint>
#include <type_traits>

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

// local headers
#include "PixelRecHitKernel.h"
#include "PixelRecHits.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  template <typename TrackerTraits>
  class setHitsLayerStart {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint32_t const* __restrict__ hitsModuleStart,
                                  pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* __restrict__ cpeParams,
                                  uint32_t* __restrict__ hitsLayerStart) const {
      ALPAKA_ASSERT_ACC(0 == hitsModuleStart[0]);

      for (int32_t i : cms::alpakatools::uniform_elements(acc, TrackerTraits::numberOfLayers + 1)) {
        hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
        int old = i == 0 ? 0 : hitsModuleStart[cpeParams->layerGeometry().layerStart[i - 1]];
        printf("LayerStart %d/%d at module %d: %d - %d\n",
               i,
               TrackerTraits::numberOfLayers,
               cpeParams->layerGeometry().layerStart[i],
               hitsLayerStart[i],
               hitsLayerStart[i] - old);
#endif
      }
    }
  };

  namespace pixelgpudetails {

    template <typename TrackerTraits>
    TrackingRecHitsSoACollection<TrackerTraits> PixelRecHitKernel<TrackerTraits>::makeHitsAsync(
        SiPixelDigisSoACollection const& digis_d,
        SiPixelClustersSoACollection const& clusters_d,
        BeamSpotPOD const* bs_d,
        pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* cpeParams,
        Queue queue) const {
      using namespace pixelRecHits;
      auto nHits = clusters_d.nClusters();
      auto offsetBPIX2 = clusters_d.offsetBPIX2();

      TrackingRecHitsSoACollection<TrackerTraits> hits_d(queue, nHits, offsetBPIX2, clusters_d->clusModuleStart());

      int activeModulesWithDigis = digis_d.nModules();

      // protect from empty events
      if (activeModulesWithDigis) {
        int threadsPerBlock = 128;
        // note: the kernel should work with an arbitrary number of blocks
        int blocks = activeModulesWithDigis;
        const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

#ifdef GPU_DEBUG
        std::cout << "launching GetHits kernel on " << alpaka::core::demangled<Acc1D> << " with " << blocks << " blocks"
                  << std::endl;
#endif
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            GetHits<TrackerTraits>{},
                            cpeParams,
                            bs_d,
                            digis_d.view(),
                            digis_d.nDigis(),
                            digis_d.nModules(),
                            clusters_d.view(),
                            hits_d.view());
#ifdef GPU_DEBUG
        alpaka::wait(queue);
#endif

        // assuming full warp of threads is better than a smaller number...
        if (nHits) {
          const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, 32);
          alpaka::exec<Acc1D>(queue,
                              workDiv1D,
                              setHitsLayerStart<TrackerTraits>{},
                              clusters_d->clusModuleStart(),
                              cpeParams,
                              hits_d.view().hitsLayerStart().data());
          constexpr auto nLayers = TrackerTraits::numberOfLayers;

          // Use a view since it's runtime sized and can't use the implicit definition
          // see HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h:100
          typename TrackingRecHitSoA<TrackerTraits>::PhiBinnerView hrv_d;
          hrv_d.assoc = &(hits_d.view().phiBinner());
          hrv_d.offSize = -1;
          hrv_d.offStorage = nullptr;
          hrv_d.contentSize = nHits;
          hrv_d.contentStorage = hits_d.view().phiBinnerStorage();

          cms::alpakatools::fillManyFromVector<Acc1D>(&(hits_d.view().phiBinner()),
                                                      hrv_d,
                                                      nLayers,
                                                      hits_d.view().iphi(),
                                                      hits_d.view().hitsLayerStart().data(),
                                                      nHits,
                                                      (uint32_t)256,
                                                      queue);

#ifdef GPU_DEBUG
          alpaka::wait(queue);
#endif
        }
      }

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "PixelRecHitKernel -> DONE!" << std::endl;
#endif

      return hits_d;
    }

    template class PixelRecHitKernel<pixelTopology::Phase1>;
    template class PixelRecHitKernel<pixelTopology::Phase2>;
    template class PixelRecHitKernel<pixelTopology::HIonPhase1>;

  }  // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
