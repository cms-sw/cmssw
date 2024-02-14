#ifndef RecoLocalTracker_SiPixelRecHits_alpaka_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_alpaka_PixelRecHits_h

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace pixelRecHits {

    template <typename TrackerTraits>
    class GetHits {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* __restrict__ cpeParams,
                                    BeamSpotPOD const* __restrict__ bs,
                                    SiPixelDigisSoAConstView digis,
                                    uint32_t numElements,
                                    SiPixelClustersSoAConstView clusters,
                                    TrackingRecHitSoAView<TrackerTraits> hits) const {
        // FIXME
        // the compiler seems NOT to optimize loads from views (even in a simple test case)
        // The whole gimnastic here of copying or not is a pure heuristic exercise that seems to produce the fastest code with the above signature
        // not using views (passing a gazzilion of array pointers) seems to produce the fastest code (but it is harder to mantain)

        ALPAKA_ASSERT_OFFLOAD(cpeParams);

        const uint32_t blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

        // copy average geometry corrected by beamspot . FIXME (move it somewhere else???)
        if (0 == blockIdx) {
          auto& agc = hits.averageGeometry();
          auto const& ag = cpeParams->averageGeometry();
          auto nLadders = TrackerTraits::numberOfLaddersInBarrel;

          cms::alpakatools::for_each_element_in_block_strided(acc, nLadders, [&](uint32_t il) {
            agc.ladderZ[il] = ag.ladderZ[il] - bs->z;
            agc.ladderX[il] = ag.ladderX[il] - bs->x;
            agc.ladderY[il] = ag.ladderY[il] - bs->y;
            agc.ladderR[il] = sqrt(agc.ladderX[il] * agc.ladderX[il] + agc.ladderY[il] * agc.ladderY[il]);
            agc.ladderMinZ[il] = ag.ladderMinZ[il] - bs->z;
            agc.ladderMaxZ[il] = ag.ladderMaxZ[il] - bs->z;
          });

          if (cms::alpakatools::once_per_block(acc)) {
            agc.endCapZ[0] = ag.endCapZ[0] - bs->z;
            agc.endCapZ[1] = ag.endCapZ[1] - bs->z;
          }
        }

        // to be moved in common namespace...
        using pixelClustering::invalidModuleId;
        constexpr int32_t MaxHitsInIter = pixelCPEforDevice::MaxHitsInIter;

        using ClusParams = pixelCPEforDevice::ClusParams;

        // as usual one block per module
        auto& clusParams = alpaka::declareSharedVar<ClusParams, __COUNTER__>(acc);

        auto me = clusters[blockIdx].moduleId();
        int nclus = clusters[me].clusInModule();

        if (0 == nclus)
          return;
#ifdef GPU_DEBUG
        if (cms::alpakatools::once_per_block(acc)) {
          auto k = clusters[1 + blockIdx].moduleStart();
          while (digis[k].moduleId() == invalidModuleId)
            ++k;
          ALPAKA_ASSERT_OFFLOAD(digis[k].moduleId() == me);
        }

        if (me % 100 == 1)
          if (cms::alpakatools::once_per_block(acc))
            printf(
                "hitbuilder: %d clusters in module %d. will write at %d\n", nclus, me, clusters[me].clusModuleStart());
#endif

        for (int startClus = 0, endClus = nclus; startClus < endClus; startClus += MaxHitsInIter) {
          auto first = clusters[1 + blockIdx].moduleStart();

          int nClusInIter = alpaka::math::min(acc, MaxHitsInIter, endClus - startClus);
          int lastClus = startClus + nClusInIter;
          assert(nClusInIter <= nclus);
          assert(nClusInIter > 0);
          assert(lastClus <= nclus);

          assert(nclus > MaxHitsInIter || (0 == startClus && nClusInIter == nclus && lastClus == nclus));

          // init
          cms::alpakatools::for_each_element_in_block_strided(acc, nClusInIter, [&](uint32_t ic) {
            clusParams.minRow[ic] = std::numeric_limits<uint32_t>::max();
            clusParams.maxRow[ic] = 0;
            clusParams.minCol[ic] = std::numeric_limits<uint32_t>::max();
            clusParams.maxCol[ic] = 0;
            clusParams.charge[ic] = 0;
            clusParams.q_f_X[ic] = 0;
            clusParams.q_l_X[ic] = 0;
            clusParams.q_f_Y[ic] = 0;
            clusParams.q_l_Y[ic] = 0;
          });

          alpaka::syncBlockThreads(acc);

          // one thread per "digi"
          const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
          const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
              cms::alpakatools::element_index_range_in_block(acc, first);
          uint32_t rowsColsFirstElementIdx = firstElementIdxNoStride;
          uint32_t rowsColsEndElementIdx = endElementIdxNoStride;
          for (uint32_t i = rowsColsFirstElementIdx; i < numElements; ++i) {
            if (not cms::alpakatools::next_valid_element_index_strided(
                    i, rowsColsFirstElementIdx, rowsColsEndElementIdx, blockDimension, numElements))
              break;
            auto id = digis[i].moduleId();
            if (id == invalidModuleId)
              continue;  // not valid
            if (id != me)
              break;  // end of module
            auto cl = digis[i].clus();
            if (cl < startClus || cl >= lastClus)
              continue;
            cl -= startClus;
            ALPAKA_ASSERT_OFFLOAD(cl >= 0);
            ALPAKA_ASSERT_OFFLOAD(cl < MaxHitsInIter);
            auto x = digis[i].xx();
            auto y = digis[i].yy();
            alpaka::atomicMin(acc, &clusParams.minRow[cl], (uint32_t)x, alpaka::hierarchy::Threads{});
            alpaka::atomicMax(acc, &clusParams.maxRow[cl], (uint32_t)x, alpaka::hierarchy::Threads{});
            alpaka::atomicMin(acc, &clusParams.minCol[cl], (uint32_t)y, alpaka::hierarchy::Threads{});
            alpaka::atomicMax(acc, &clusParams.maxCol[cl], (uint32_t)y, alpaka::hierarchy::Threads{});
          }

          alpaka::syncBlockThreads(acc);

          auto pixmx = cpeParams->detParams(me).pixmx;
          uint32_t chargeFirstElementIdx = firstElementIdxNoStride;
          uint32_t chargeEndElementIdx = endElementIdxNoStride;
          for (uint32_t i = chargeFirstElementIdx; i < numElements; ++i) {
            if (not cms::alpakatools::next_valid_element_index_strided(
                    i, chargeFirstElementIdx, chargeEndElementIdx, blockDimension, numElements))
              break;
            auto id = digis[i].moduleId();
            if (id == invalidModuleId)
              continue;  // not valid
            if (id != me)
              break;  // end of module
            auto cl = digis[i].clus();
            if (cl < startClus || cl >= lastClus)
              continue;
            cl -= startClus;
            ALPAKA_ASSERT_OFFLOAD(cl >= 0);
            ALPAKA_ASSERT_OFFLOAD(cl < MaxHitsInIter);
            auto x = digis[i].xx();
            auto y = digis[i].yy();
            auto ch = digis[i].adc();
            alpaka::atomicAdd(acc, &clusParams.charge[cl], (int32_t)ch, alpaka::hierarchy::Threads{});
            ch = alpaka::math::min(acc, ch, pixmx);
            if (clusParams.minRow[cl] == x)
              alpaka::atomicAdd(acc, &clusParams.q_f_X[cl], (int32_t)ch, alpaka::hierarchy::Threads{});
            if (clusParams.maxRow[cl] == x)
              alpaka::atomicAdd(acc, &clusParams.q_l_X[cl], (int32_t)ch, alpaka::hierarchy::Threads{});
            if (clusParams.minCol[cl] == y)
              alpaka::atomicAdd(acc, &clusParams.q_f_Y[cl], (int32_t)ch, alpaka::hierarchy::Threads{});
            if (clusParams.maxCol[cl] == y)
              alpaka::atomicAdd(acc, &clusParams.q_l_Y[cl], (int32_t)ch, alpaka::hierarchy::Threads{});
          }

          alpaka::syncBlockThreads(acc);

          // next one cluster per thread...
          first = clusters[me].clusModuleStart() + startClus;
          cms::alpakatools::for_each_element_in_block_strided(acc, nClusInIter, [&](uint32_t ic) {
            auto h = first + ic;  // output index in global memory

            assert(h < (uint32_t)hits.metadata().size());
            assert(h < clusters[me + 1].clusModuleStart());

            pixelCPEforDevice::position<TrackerTraits>(
                cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);

            pixelCPEforDevice::errorFromDB<TrackerTraits>(
                cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);

            // store it
            hits[h].chargeAndStatus().charge = clusParams.charge[ic];
            hits[h].chargeAndStatus().status = clusParams.status[ic];
            hits[h].detectorIndex() = me;

            float xl, yl;
            hits[h].xLocal() = xl = clusParams.xpos[ic];
            hits[h].yLocal() = yl = clusParams.ypos[ic];

            hits[h].clusterSizeX() = clusParams.xsize[ic];
            hits[h].clusterSizeY() = clusParams.ysize[ic];

            hits[h].xerrLocal() = clusParams.xerr[ic] * clusParams.xerr[ic] + cpeParams->detParams(me).apeXX;
            hits[h].yerrLocal() = clusParams.yerr[ic] * clusParams.yerr[ic] + cpeParams->detParams(me).apeYY;

            // keep it local for computations
            float xg, yg, zg;
            // to global and compute phi...
            cpeParams->detParams(me).frame.toGlobal(xl, yl, xg, yg, zg);
            // here correct for the beamspot...
            xg -= bs->x;
            yg -= bs->y;
            zg -= bs->z;

            hits[h].xGlobal() = xg;
            hits[h].yGlobal() = yg;
            hits[h].zGlobal() = zg;

            hits[h].rGlobal() = alpaka::math::sqrt(acc, xg * xg + yg * yg);
            hits[h].iphi() = unsafe_atan2s<7>(yg, xg);
          });
          alpaka::syncBlockThreads(acc);
        }  // end loop on batches
      }
    };

  }  // namespace pixelRecHits
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_alpaka_PixelRecHits_h
