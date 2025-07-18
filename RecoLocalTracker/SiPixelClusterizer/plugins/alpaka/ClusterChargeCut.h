#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_ClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_ClusterChargeCut_h

#include <cstdint>
#include <cstdio>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"

//#define GPU_DEBUG

namespace pixelClustering {

  template <typename TrackerTraits>
  struct ClusterChargeCut {
    template <typename TAcc>
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
        }

#ifdef GPU_DEBUG
        if (thisModuleId % 100 == 1)
          if (cms::alpakatools::once_per_block(acc))
            printf("start cluster charge cut for module %d in block %d\n", thisModuleId, module);
#endif

        ALPAKA_ASSERT_ACC(nclus <= maxNumClustersPerModules);
        for (auto i : cms::alpakatools::independent_group_elements(acc, nclus)) {
          charge[i] = 0;
        }
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
        }
        alpaka::syncBlockThreads(acc);

        auto chargeCut = clusterThresholds.getThresholdForLayerOnCondition(thisModuleId < startBPIX2);

        bool good = true;
        for (auto i : cms::alpakatools::independent_group_elements(acc, nclus)) {
          newclusId[i] = ok[i] = (charge[i] >= chargeCut) ? 1 : 0;
          if (0 == ok[i])
            good = false;
#ifdef GPU_DEBUG
          printf("Cutting pix %d in module %d newId %d ok? %d charge %d cut %d -> good %d \n",
                 i,
                 thisModuleId,
                 newclusId[i],
                 ok[i],
                 charge[i],
                 chargeCut,
                 good);
#endif
        }
        // if all clusters are above threshold, do nothing
        if (alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, good))
          continue;

        // renumber
        // FIXME move this logic inside a single prefixscan() function ?
        if constexpr (cms::alpakatools::requires_single_thread_per_block_v<TAcc>) {
          // for a single-threaded accelerator, use a simple loop
          for (uint32_t i = 1; i < nclus; ++i) {
            newclusId[i] += newclusId[i - 1];
          }
        } else {
          // for a multi-threaded accelerator, use an iterative block-based prefix scan
          constexpr int warpSize = cms::alpakatools::warpSize;
          // FIXME this value should come from cms::alpakatools::blockPrefixScan itself
          constexpr uint32_t maxThreads = warpSize * warpSize;

          auto& ws = alpaka::declareSharedVar<uint16_t[warpSize], __COUNTER__>(acc);
          auto minClust = std::min(nclus, maxThreads);

          // process the first maxThreads elements
          cms::alpakatools::blockPrefixScan(acc, newclusId, minClust, ws);

          // if there may be more than maxThreads elements, repeat the prefix scan and update the intermediat sums
          if constexpr (maxNumClustersPerModules > maxThreads) {
            for (uint32_t offset = maxThreads; offset < nclus; offset += maxThreads) {
              cms::alpakatools::blockPrefixScan(acc, newclusId + offset, nclus - offset, ws);
              for (uint32_t i : cms::alpakatools::independent_group_elements(acc, offset, nclus)) {
                uint32_t prevBlockEnd = (i / maxThreads) * maxThreads - 1;
                newclusId[i] += newclusId[prevBlockEnd];
              }
              alpaka::syncBlockThreads(acc);
            }
          }
        }

        ALPAKA_ASSERT_ACC(nclus >= newclusId[nclus - 1]);

        clus_view[thisModuleId].clusInModule() = newclusId[nclus - 1];

        // reassign id
        for (auto i : cms::alpakatools::independent_group_elements(acc, firstPixel, numElements)) {
          if (digi_view[i].moduleId() == invalidModuleId)
            continue;  // not valid
          if (digi_view[i].moduleId() != thisModuleId)
            break;  // end of module
          if (0 == ok[digi_view[i].clus()])
            digi_view[i].moduleId() = digi_view[i].clus() = invalidModuleId;
          else
            digi_view[i].clus() = newclusId[digi_view[i].clus()] - 1;
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
