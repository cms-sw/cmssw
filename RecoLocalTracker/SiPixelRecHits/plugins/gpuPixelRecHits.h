#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h

#include <cstdint>
#include <cstdio>
#include <limits>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

namespace gpuPixelRecHits {

  __global__ void getHits(pixelCPEforGPU::ParamsOnGPU const* __restrict__ cpeParams,
                          BeamSpotCUDA::Data const* __restrict__ bs,
                          SiPixelDigisCUDA::DeviceConstView const* __restrict__ pdigis,
                          int numElements,
                          SiPixelClustersCUDA::DeviceConstView const* __restrict__ pclusters,
                          TrackingRecHit2DSOAView* phits) {
    // FIXME
    // the compiler seems NOT to optimize loads from views (even in a simple test case)
    // The whole gimnastic here of copying or not is a pure heuristic exercise that seems to produce the fastest code with the above signature
    // not using views (passing a gazzilion of array pointers) seems to produce the fastest code (but it is harder to mantain)

    assert(phits);
    assert(cpeParams);

    auto& hits = *phits;

    auto const digis = *pdigis;  // the copy is intentional!
    auto const& clusters = *pclusters;

    // copy average geometry corrected by beamspot . FIXME (move it somewhere else???)
    if (0 == blockIdx.x) {
      auto& agc = hits.averageGeometry();
      auto const& ag = cpeParams->averageGeometry();
      for (int il = threadIdx.x, nl = TrackingRecHit2DSOAView::AverageGeometry::numberOfLaddersInBarrel; il < nl;
           il += blockDim.x) {
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
        //         printf("endcapZ %f %f\n",agc.endCapZ[0],agc.endCapZ[1]);
      }
    }

    // to be moved in common namespace...
    constexpr uint16_t InvId = 9999;  // must be > MaxNumModules
    constexpr int32_t MaxHitsInIter = pixelCPEforGPU::MaxHitsInIter;

    using ClusParams = pixelCPEforGPU::ClusParams;

    // as usual one block per module
    __shared__ ClusParams clusParams;

    auto me = clusters.moduleId(blockIdx.x);
    int nclus = clusters.clusInModule(me);

    if (0 == nclus)
      return;

#ifdef GPU_DEBUG
    if (threadIdx.x == 0) {
      auto k = first;
      while (digis.moduleInd(k) == InvId)
        ++k;
      assert(digis.moduleInd(k) == me);
    }
#endif

#ifdef GPU_DEBUG
    if (me % 100 == 1)
      if (threadIdx.x == 0)
        printf("hitbuilder: %d clusters in module %d. will write at %d\n", nclus, me, clusters.clusModuleStart(me));
#endif

    for (int startClus = 0, endClus = nclus; startClus < endClus; startClus += MaxHitsInIter) {
      auto first = clusters.moduleStart(1 + blockIdx.x);

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
        clusParams.Q_f_X[ic] = 0;
        clusParams.Q_l_X[ic] = 0;
        clusParams.Q_f_Y[ic] = 0;
        clusParams.Q_l_Y[ic] = 0;
      }

      first += threadIdx.x;

      __syncthreads();

      // one thead per "digi"

      for (int i = first; i < numElements; i += blockDim.x) {
        auto id = digis.moduleInd(i);
        if (id == InvId)
          continue;  // not valid
        if (id != me)
          break;  // end of module
        auto cl = digis.clus(i);
        if (cl < startClus || cl >= lastClus)
          continue;
        auto x = digis.xx(i);
        auto y = digis.yy(i);
        cl -= startClus;
        assert(cl >= 0);
        assert(cl < MaxHitsInIter);
        atomicMin(&clusParams.minRow[cl], x);
        atomicMax(&clusParams.maxRow[cl], x);
        atomicMin(&clusParams.minCol[cl], y);
        atomicMax(&clusParams.maxCol[cl], y);
      }

      __syncthreads();

      for (int i = first; i < numElements; i += blockDim.x) {
        auto id = digis.moduleInd(i);
        if (id == InvId)
          continue;  // not valid
        if (id != me)
          break;  // end of module
        auto cl = digis.clus(i);
        if (cl < startClus || cl >= lastClus)
          continue;
        cl -= startClus;
        assert(cl >= 0);
        assert(cl < MaxHitsInIter);
        auto x = digis.xx(i);
        auto y = digis.yy(i);
        auto ch = digis.adc(i);
        atomicAdd(&clusParams.charge[cl], ch);
        if (clusParams.minRow[cl] == x)
          atomicAdd(&clusParams.Q_f_X[cl], ch);
        if (clusParams.maxRow[cl] == x)
          atomicAdd(&clusParams.Q_l_X[cl], ch);
        if (clusParams.minCol[cl] == y)
          atomicAdd(&clusParams.Q_f_Y[cl], ch);
        if (clusParams.maxCol[cl] == y)
          atomicAdd(&clusParams.Q_l_Y[cl], ch);
      }

      __syncthreads();

      // next one cluster per thread...

      first = clusters.clusModuleStart(me) + startClus;

      for (int ic = threadIdx.x; ic < nClusInIter; ic += blockDim.x) {
        auto h = first + ic;  // output index in global memory

        // this cannot happen anymore
        if (h >= TrackingRecHit2DSOAView::maxHits())
          break;  // overflow...
        assert(h < hits.nHits());
        assert(h < clusters.clusModuleStart(me + 1));

        pixelCPEforGPU::position(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);
        pixelCPEforGPU::errorFromDB(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);

        // store it

        hits.charge(h) = clusParams.charge[ic];

        hits.detectorIndex(h) = me;

        float xl, yl;
        hits.xLocal(h) = xl = clusParams.xpos[ic];
        hits.yLocal(h) = yl = clusParams.ypos[ic];

        hits.clusterSizeX(h) = clusParams.xsize[ic];
        hits.clusterSizeY(h) = clusParams.ysize[ic];

        hits.xerrLocal(h) = clusParams.xerr[ic] * clusParams.xerr[ic];
        hits.yerrLocal(h) = clusParams.yerr[ic] * clusParams.yerr[ic];

        // keep it local for computations
        float xg, yg, zg;
        // to global and compute phi...
        cpeParams->detParams(me).frame.toGlobal(xl, yl, xg, yg, zg);
        // here correct for the beamspot...
        xg -= bs->x;
        yg -= bs->y;
        zg -= bs->z;

        hits.xGlobal(h) = xg;
        hits.yGlobal(h) = yg;
        hits.zGlobal(h) = zg;

        hits.rGlobal(h) = std::sqrt(xg * xg + yg * yg);
        hits.iphi(h) = unsafe_atan2s<7>(yg, xg);
      }
      __syncthreads();
    }  // end loop on batches
  }

}  // namespace gpuPixelRecHits

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
