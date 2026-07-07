#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_ClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_ClusterChargeCut_h

#include <cstdint>
#include <cstdio>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"

//#define GPU_DEBUG

namespace pixelClustering {

  template <typename TrackerTraits>
  struct ClusterChargeCut {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiPixelDigisSoAView digi_view,
                                  SiPixelClustersSoAView clus_view,
                                  // charge cut on cluster in electrons (for layer 1 and for other layers)
                                  SiPixelClusterThresholds clusterThresholds,
                                  const uint32_t numElements) const {
      constexpr int32_t maxNumClustersPerModules = TrackerTraits::maxNumClustersPerModules;

#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("All digis before cut: \n");
        for (uint32_t i = 0; i < numElements; i++)
          printf("%d %d %d %d %d \n",
                 i,
                 digi_view[i].rawIdArr(),
                 digi_view[i].clus(),
                 digi_view[i].pdigi(),
                 digi_view[i].adc());
      }
      alpaka::syncBlockThreads(acc);
#endif

      auto& charge = alpaka::declareSharedVar<int32_t[maxNumClustersPerModules], __COUNTER__>(acc);
      auto& ok = alpaka::declareSharedVar<uint8_t[maxNumClustersPerModules], __COUNTER__>(acc);
      auto& newclusId = alpaka::declareSharedVar<uint16_t[maxNumClustersPerModules], __COUNTER__>(acc);
      // per-cluster minimum packed pixel coordinate (row << 16 | col): a physical, run-independent
      // key that is unique within the module (a pixel belongs to exactly one cluster), used to
      // renumber the clusters deterministically (see below)
      auto& minPix = alpaka::declareSharedVar<uint32_t[maxNumClustersPerModules], __COUNTER__>(acc);
      auto& nOk = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);

      constexpr int startBPIX2 = TrackerTraits::layerStart[1];

      ALPAKA_ASSERT_ACC(TrackerTraits::numberOfModules < maxNumModules);
      ALPAKA_ASSERT_ACC(startBPIX2 < TrackerTraits::numberOfModules);

      auto endModule = clus_view[0].moduleStart();

      for (auto module : cms::alpakatools::independent_groups(acc, endModule)) {
        auto firstPixel = clus_view[1 + module].moduleStart();
        auto thisModuleId = digi_view[firstPixel].moduleId();
        while (thisModuleId == invalidModuleId and firstPixel < numElements) {
          // skip invalid or duplicate pixels
          ++firstPixel;
          thisModuleId = digi_view[firstPixel].moduleId();
        }
        if (firstPixel >= numElements) {
          // reached the end of the input while skipping the invalid pixels, nothing left to do
          break;
        }
        if (thisModuleId != clus_view[module].moduleId()) {
          // reached the end of the module while skipping the invalid pixels, skip this module
          continue;
        }
        ALPAKA_ASSERT_ACC(thisModuleId < TrackerTraits::numberOfModules);

        uint32_t nclus = clus_view[thisModuleId].clusInModule();
        if (nclus == 0)
          return;

        if (cms::alpakatools::once_per_block(acc) && nclus > maxNumClustersPerModules)
          printf("Warning: too many clusters in module %u in block %u: %u > %d\n",
                 thisModuleId,
                 module,
                 nclus,
                 maxNumClustersPerModules);

        if (nclus > maxNumClustersPerModules) {
          // remove excess  FIXME find a way to cut charge first....
          for (auto i : cms::alpakatools::independent_group_elements(acc, firstPixel, numElements)) {
            if (digi_view[i].moduleId() == invalidModuleId)
              continue;  // not valid
            if (digi_view[i].moduleId() != thisModuleId)
              break;  // end of module
            if (digi_view[i].clus() >= maxNumClustersPerModules) {
              digi_view[i].moduleId() = invalidModuleId;
              digi_view[i].clus() = invalidModuleId;
            }
          }
          nclus = maxNumClustersPerModules;
          clus_view[thisModuleId].clusInModule() = nclus;
        }

        ALPAKA_ASSERT_ACC(clus_view[thisModuleId].clusInModule() <= maxNumClustersPerModules);

#ifdef GPU_DEBUG
        if (thisModuleId % 100 == 1)
          if (cms::alpakatools::once_per_block(acc))
            printf("start cluster charge cut for module %d in block %d\n", thisModuleId, module);
#endif

        ALPAKA_ASSERT_ACC(nclus <= maxNumClustersPerModules);
        for (auto i : cms::alpakatools::independent_group_elements(acc, nclus)) {
          charge[i] = 0;
          minPix[i] = std::numeric_limits<uint32_t>::max();
        }
        if (cms::alpakatools::once_per_block(acc))
          nOk = 0;
        alpaka::syncBlockThreads(acc);

        for (auto i : cms::alpakatools::independent_group_elements(acc, firstPixel, numElements)) {
          if (digi_view[i].moduleId() == invalidModuleId)
            continue;  // not valid
          if (digi_view[i].moduleId() != thisModuleId)
            break;  // end of module
          alpaka::atomicAdd(acc,
                            &charge[digi_view[i].clus()],
                            static_cast<int32_t>(digi_view[i].adc()),
                            alpaka::hierarchy::Threads{});
          alpaka::atomicMin(acc,
                            &minPix[digi_view[i].clus()],
                            (uint32_t(digi_view[i].xx()) << 16) | digi_view[i].yy(),
                            alpaka::hierarchy::Threads{});
        }
        alpaka::syncBlockThreads(acc);

        auto chargeCut = clusterThresholds.getThresholdForLayerOnCondition(thisModuleId < startBPIX2);

        for (auto i : cms::alpakatools::independent_group_elements(acc, nclus)) {
          ok[i] = (charge[i] >= chargeCut) ? 1 : 0;
          if (ok[i])
            alpaka::atomicAdd(acc, &nOk, 1u, alpaka::hierarchy::Threads{});
#ifdef GPU_DEBUG
          printf("Cutting pix %d in module %d ok? %d charge %d cut %d\n", i, thisModuleId, ok[i], charge[i], chargeCut);
#endif
        }
        alpaka::syncBlockThreads(acc);

        // Renumber the surviving clusters by ascending minimum pixel coordinate.
        // The clusters are labelled by FindClus in atomic (run-dependent) order, and the cluster id
        // determines the position of the corresponding RecHit in the hit SoA: renumbering by a
        // physical key makes the cluster ids - and everything built on top of them - reproducible
        // run-to-run and across backends. The key is unique within the module, so the ranks form a
        // permutation of [0, nOk). The O(nclus^2) rank-by-counting is cheap: at PU200 a module has
        // ~45 clusters on average and less than ~200 at the 99.9% quantile.
        for (auto i : cms::alpakatools::independent_group_elements(acc, nclus)) {
          if (!ok[i])
            continue;
          auto const key = minPix[i];
          uint16_t rank = 0;
          for (uint32_t j = 0; j < nclus; ++j)
            rank += (ok[j] && (minPix[j] < key)) ? 1 : 0;
          newclusId[i] = rank;
          ALPAKA_ASSERT_ACC(rank < nOk);
        }
        alpaka::syncBlockThreads(acc);

        clus_view[thisModuleId].clusInModule() = nOk;

        // reassign id
        for (auto i : cms::alpakatools::independent_group_elements(acc, firstPixel, numElements)) {
          if (digi_view[i].moduleId() == invalidModuleId)
            continue;  // not valid
          if (digi_view[i].moduleId() != thisModuleId)
            break;  // end of module
          if (0 == ok[digi_view[i].clus()])
            digi_view[i].moduleId() = digi_view[i].clus() = invalidModuleId;
          else
            digi_view[i].clus() = newclusId[digi_view[i].clus()];
        }

        // done
        alpaka::syncBlockThreads(acc);
#ifdef GPU_DEBUG
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("All digis AFTER cut: \n");
          for (uint32_t i = 0; i < numElements; i++)
            printf("%d %d %d %d %d \n",
                   i,
                   digi_view[i].rawIdArr(),
                   digi_view[i].clus(),
                   digi_view[i].pdigi(),
                   digi_view[i].adc());
        }
#endif
      }
    }
  };

}  // namespace pixelClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_ClusterChargeCut_h
