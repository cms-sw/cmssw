#ifndef RecoLocalTracker_SiPixelClusterizer_alpaka_ClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_alpaka_ClusterChargeCut_h

#include <cstdint>
#include <cstdio>

#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"

//#define GPU_DEBUG

namespace pixelClustering {

  template <typename TrackerTraits>
  struct ClusterChargeCut {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc,
        SiPixelDigisSoAView digi_view,
        SiPixelClustersSoAView clus_view,
        SiPixelClusterThresholds
            clusterThresholds,  // charge cut on cluster in electrons (for layer 1 and for other layers)
        const uint32_t numElements) const {
      constexpr int startBPIX2 = TrackerTraits::layerStart[1];
      constexpr int32_t maxNumClustersPerModules = TrackerTraits::maxNumClustersPerModules;
      [[maybe_unused]] constexpr int nMaxModules = TrackerTraits::numberOfModules;

      const uint32_t blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      auto firstModule = blockIdx;
      auto endModule = clus_view[0].moduleStart();
      if (blockIdx >= endModule)
        return;

      auto& charge = alpaka::declareSharedVar<int32_t[maxNumClustersPerModules], __COUNTER__>(acc);
      auto& ok = alpaka::declareSharedVar<uint8_t[maxNumClustersPerModules], __COUNTER__>(acc);
      auto& newclusId = alpaka::declareSharedVar<uint16_t[maxNumClustersPerModules], __COUNTER__>(acc);

      const uint32_t gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

      for (auto module = firstModule; module < endModule; module += gridDimension) {
        auto firstPixel = clus_view[1 + module].moduleStart();
        auto thisModuleId = digi_view[firstPixel].moduleId();

        ALPAKA_ASSERT_OFFLOAD(nMaxModules < maxNumModules);
        ALPAKA_ASSERT_OFFLOAD(startBPIX2 < nMaxModules);

        uint32_t nclus = clus_view[thisModuleId].clusInModule();
        if (nclus == 0)
          return;

        if (cms::alpakatools::once_per_block(acc) && nclus > maxNumClustersPerModules)
          printf("Warning too many clusters in module %d in block %d: %d > %d\n",
                 thisModuleId,
                 module,
                 nclus,
                 maxNumClustersPerModules);

        // Stride = block size.
        const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

        // Get thread / CPU element indices in block.
        const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
            cms::alpakatools::element_index_range_in_block(acc, firstPixel);

        if (nclus > maxNumClustersPerModules) {
          uint32_t firstElementIdx = firstElementIdxNoStride;
          uint32_t endElementIdx = endElementIdxNoStride;
          // remove excess  FIXME find a way to cut charge first....
          for (uint32_t i = firstElementIdx; i < numElements; ++i) {
            if (not cms::alpakatools::next_valid_element_index_strided(
                    i, firstElementIdx, endElementIdx, blockDimension, numElements))
              break;
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

        ALPAKA_ASSERT_OFFLOAD(nclus <= maxNumClustersPerModules);
        cms::alpakatools::for_each_element_in_block_strided(acc, nclus, [&](uint32_t i) { charge[i] = 0; });
        alpaka::syncBlockThreads(acc);

        uint32_t firstElementIdx = firstElementIdxNoStride;
        uint32_t endElementIdx = endElementIdxNoStride;
        for (uint32_t i = firstElementIdx; i < numElements; ++i) {
          if (not cms::alpakatools::next_valid_element_index_strided(
                  i, firstElementIdx, endElementIdx, blockDimension, numElements))
            break;
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
        bool allGood = true;

        cms::alpakatools::for_each_element_in_block_strided(acc, nclus, [&](uint32_t i) {
          newclusId[i] = ok[i] = (charge[i] > chargeCut) ? 1 : 0;
          if (ok[i] == 0)
            allGood = allGood && false;

          // #ifdef GPU_DEBUG
          // printf("module %d -> chargeCut = %d; cluster %d; charge = %d; ok = %s\n",thisModuleId, chargeCut,i,charge[i],ok[i] > 0 ? " -> good" : "-> cut");
          // #endif
        });
        alpaka::syncBlockThreads(acc);

        // if all clusters above threshold do nothing
        // if (allGood)
        //   continue;

        // renumber
        auto& ws = alpaka::declareSharedVar<uint16_t[32], __COUNTER__>(acc);
        constexpr uint32_t maxThreads = 1024;
        auto minClust = std::min(nclus, maxThreads);

        cms::alpakatools::blockPrefixScan(acc, newclusId, minClust, ws);

        if constexpr (maxNumClustersPerModules > maxThreads)  //only if needed
        {
          for (uint32_t offset = maxThreads; offset < nclus; offset += maxThreads) {
            cms::alpakatools::blockPrefixScan(acc, newclusId + offset, nclus - offset, ws);

            cms::alpakatools::for_each_element_in_block_strided(acc, nclus - offset, [&](uint32_t i) {
              uint32_t prevBlockEnd = ((i + offset / maxThreads) * maxThreads) - 1;
              newclusId[i] += newclusId[prevBlockEnd];
            });
            alpaka::syncBlockThreads(acc);
          }
        }

        ALPAKA_ASSERT_OFFLOAD(nclus >= newclusId[nclus - 1]);

        if (nclus == newclusId[nclus - 1])
          return;

        clus_view[thisModuleId].clusInModule() = newclusId[nclus - 1];
        alpaka::syncBlockThreads(acc);

#ifdef GPU_DEBUG
        if (thisModuleId % 100 == 1)
          if (cms::alpakatools::once_per_block(acc))
            printf("module %d -> chargeCut = %d; nclus (pre cut) = %d; nclus (after cut) = %d\n",
                   thisModuleId,
                   chargeCut,
                   nclus,
                   clus_view[thisModuleId].clusInModule());
#endif
        // mark bad cluster again
        cms::alpakatools::for_each_element_in_block_strided(acc, nclus, [&](uint32_t i) {
          if (0 == ok[i])
            newclusId[i] = invalidModuleId + 1;
        });

        alpaka::syncBlockThreads(acc);

        // reassign id
        firstElementIdx = firstElementIdxNoStride;
        endElementIdx = endElementIdxNoStride;
        for (uint32_t i = firstElementIdx; i < numElements; ++i) {
          if (not cms::alpakatools::next_valid_element_index_strided(
                  i, firstElementIdx, endElementIdx, blockDimension, numElements))
            break;
          if (digi_view[i].moduleId() == invalidModuleId)
            continue;  // not valid
          if (digi_view[i].moduleId() != thisModuleId)
            break;  // end of module
          if (0 == ok[digi_view[i].clus()])
            digi_view[i].moduleId() = digi_view[i].clus() = invalidModuleId;
          else
            digi_view[i].clus() = newclusId[digi_view[i].clus()] - 1;
          // digi_view[i].clus() = newclusId[digi_view[i].clus()] - 1;
          // if (digi_view[i].clus() == invalidModuleId)
          //   digi_view[i].moduleId() = invalidModuleId;
        }

        alpaka::syncBlockThreads(acc);

        //done
      }
    }
  };

}  // namespace pixelClustering

#endif  //
