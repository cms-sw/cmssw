#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h

#include <cstdint>
#include <cstdio>
#include <limits>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"

//#define GPU_DEBUG 1
namespace gpuPixelRecHits {

  template <typename TrackerTraits>
  __global__ void getHits(pixelCPEforGPU::ParamsOnGPUT<TrackerTraits> const* __restrict__ cpeParams,
                          BeamSpotPOD const* __restrict__ bs,
                          SiPixelDigisCUDASOAConstView digis,
                          int numElements,
                          SiPixelClustersCUDASOAConstView clusters,
                          TrackingRecHitSoAView<TrackerTraits> hits) {
    // FIXME
    // the compiler seems NOT to optimize loads from views (even in a simple test case)
    // The whole gimnastic here of copying or not is a pure heuristic exercise that seems to produce the fastest code with the above signature
    // not using views (passing a gazzilion of array pointers) seems to produce the fastest code (but it is harder to mantain)

    assert(cpeParams);

    // copy average geometry corrected by beamspot . FIXME (move it somewhere else???)
    if (0 == blockIdx.x) {
      auto& agc = hits.averageGeometry();
      auto const& ag = cpeParams->averageGeometry();
      auto nLadders = TrackerTraits::numberOfLaddersInBarrel;

      for (int il = threadIdx.x, nl = nLadders; il < nl; il += blockDim.x) {
        agc.ladderZ[il] = ag.ladderZ[il] - bs->z;
        agc.ladderX[il] = ag.ladderX[il] - bs->x;
        agc.ladderY[il] = ag.ladderY[il] - bs->y;
        agc.ladderR[il] = sqrt(agc.ladderX[il] * agc.ladderX[il] + agc.ladderY[il] * agc.ladderY[il]);
        agc.ladderMinZ[il] = ag.ladderMinZ[il] - bs->z;
        agc.ladderMaxZ[il] = ag.ladderMaxZ[il] - bs->z;
      }

      if (0 == threadIdx.x) {
        agc.endCapZ[0] = ag.endCapZ[0] - bs->z;
        agc.endCapZ[1] = ag.endCapZ[1] - bs->z;
      }
    }

    // to be moved in common namespace...
    using gpuClustering::invalidModuleId;
    constexpr int32_t MaxHitsInIter = pixelCPEforGPU::MaxHitsInIter;

    using ClusParams = pixelCPEforGPU::ClusParams;

    // as usual one block per module
    __shared__ ClusParams clusParams;

    auto me = clusters[blockIdx.x].moduleId();
    int nclus = clusters[me].clusInModule();

    if (0 == nclus)
      return;
#ifdef GPU_DEBUG
    if (threadIdx.x == 0) {
      auto k = clusters[1 + blockIdx.x].moduleStart();
      while (digis[k].moduleId() == invalidModuleId)
        ++k;
      assert(digis[k].moduleId() == me);
    }

    if (me % 100 == 1)
      if (threadIdx.x == 0)
        printf("hitbuilder: %d clusters in module %d. will write at %d\n", nclus, me, clusters[me].clusModuleStart());
#endif

    for (int startClus = 0, endClus = nclus; startClus < endClus; startClus += MaxHitsInIter) {
      int nClusInIter = std::min(MaxHitsInIter, endClus - startClus);
      int lastClus = startClus + nClusInIter;
      assert(nClusInIter <= nclus);
      assert(nClusInIter > 0);
      assert(lastClus <= nclus);

      assert(nclus > MaxHitsInIter || (0 == startClus && nClusInIter == nclus && lastClus == nclus));

      // init
      for (int ic = threadIdx.x; ic < nClusInIter; ic += blockDim.x) {
        clusParams.minRow[ic] = std::numeric_limits<uint32_t>::max();
        clusParams.maxRow[ic] = 0;
        clusParams.minCol[ic] = std::numeric_limits<uint32_t>::max();
        clusParams.maxCol[ic] = 0;
        clusParams.charge[ic] = 0;
        clusParams.q_f_X[ic] = 0;
        clusParams.q_l_X[ic] = 0;
        clusParams.q_f_Y[ic] = 0;
        clusParams.q_l_Y[ic] = 0;
      }

      __syncthreads();

      // one thread per "digi"
      auto first = clusters[1 + blockIdx.x].moduleStart() + threadIdx.x;
      for (int i = first; i < numElements; i += blockDim.x) {
        auto id = digis[i].moduleId();
        if (id == invalidModuleId)
          continue;  // not valid
        if (id != me)
          break;  // end of module
        auto cl = digis[i].clus();
        if (cl < startClus || cl >= lastClus)
          continue;
        cl -= startClus;
        assert(cl >= 0);
        assert(cl < MaxHitsInIter);
        auto x = digis[i].xx();
        auto y = digis[i].yy();
        atomicMin(&clusParams.minRow[cl], x);
        atomicMax(&clusParams.maxRow[cl], x);
        atomicMin(&clusParams.minCol[cl], y);
        atomicMax(&clusParams.maxCol[cl], y);
      }

      __syncthreads();

      auto pixmx = cpeParams->detParams(me).pixmx;
      for (int i = first; i < numElements; i += blockDim.x) {
        auto id = digis[i].moduleId();
        if (id == invalidModuleId)
          continue;  // not valid
        if (id != me)
          break;  // end of module
        auto cl = digis[i].clus();
        if (cl < startClus || cl >= lastClus)
          continue;
        cl -= startClus;
        assert(cl >= 0);
        assert(cl < MaxHitsInIter);
        auto x = digis[i].xx();
        auto y = digis[i].yy();
        auto ch = digis[i].adc();
        atomicAdd(&clusParams.charge[cl], ch);
        ch = std::min(ch, pixmx);
        if (clusParams.minRow[cl] == x)
          atomicAdd(&clusParams.q_f_X[cl], ch);
        if (clusParams.maxRow[cl] == x)
          atomicAdd(&clusParams.q_l_X[cl], ch);
        if (clusParams.minCol[cl] == y)
          atomicAdd(&clusParams.q_f_Y[cl], ch);
        if (clusParams.maxCol[cl] == y)
          atomicAdd(&clusParams.q_l_Y[cl], ch);
      }

      __syncthreads();

      // next one cluster per thread...

      first = clusters[me].clusModuleStart() + startClus;
      for (int ic = threadIdx.x; ic < nClusInIter; ic += blockDim.x) {
        auto h = first + ic;  // output index in global memory

        assert(h < hits.nHits());
        assert(h < clusters[me + 1].clusModuleStart());

        pixelCPEforGPU::position<TrackerTraits>(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);

        pixelCPEforGPU::errorFromDB<TrackerTraits>(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);

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

        hits[h].rGlobal() = std::sqrt(xg * xg + yg * yg);
        hits[h].iphi() = unsafe_atan2s<7>(yg, xg);
      }
      __syncthreads();
    }  // end loop on batches
  }

}  // namespace gpuPixelRecHits

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
