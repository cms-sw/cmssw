#ifndef RecoPixelVertexing_PixelVertexFinding_plugins_gpuClusterTracksByDensity_h
#define RecoPixelVertexing_PixelVertexFinding_plugins_gpuClusterTracksByDensity_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  // this algo does not really scale as it works in a single block...
  // enough for <10K tracks we have
  //
  // based on Rodrighez&Laio algo
  //
  __device__ __forceinline__ void clusterTracksByDensity(VtxSoAView& pdata,
                                                         WsSoAView& pws,
                                                         int minT,      // min number of neighbours to be "seed"
                                                         float eps,     // max absolute distance to cluster
                                                         float errmax,  // max error to be "seed"
                                                         float chi2max  // max normalized distance to cluster
  ) {
    using namespace gpuVertexFinder;
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    if (verbose && 0 == threadIdx.x)
      printf("params %d %f %f %f\n", minT, eps, errmax, chi2max);

    auto er2mx = errmax * errmax;

    auto& __restrict__ data = pdata;
    auto& __restrict__ ws = pws;
    auto nt = ws.ntrks();
    float const* __restrict__ zt = ws.zt();
    float const* __restrict__ ezt2 = ws.ezt2();

    uint32_t& nvFinal = data.nvFinal();
    uint32_t& nvIntermediate = ws.nvIntermediate();

    uint8_t* __restrict__ izt = ws.izt();
    int32_t* __restrict__ nn = data.ndof();
    int32_t* __restrict__ iv = ws.iv();

    assert(zt);
    assert(ezt2);
    assert(izt);
    assert(nn);
    assert(iv);

    using Hist = cms::cuda::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;
    __shared__ Hist hist;
    __shared__ typename Hist::Counter hws[32];
    for (auto j = threadIdx.x; j < Hist::totbins(); j += blockDim.x) {
      hist.off[j] = 0;
    }
    __syncthreads();

    if (verbose && 0 == threadIdx.x)
      printf("booked hist with %d bins, size %d for %d tracks\n", hist.nbins(), hist.capacity(), nt);

    assert((int)nt <= hist.capacity());

    // fill hist  (bin shall be wider than "eps")
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      assert(i < zVertex::utilities::MAXTRACKS);
      int iz = int(zt[i] * 10.);  // valid if eps<=0.1
      // iz = std::clamp(iz, INT8_MIN, INT8_MAX);  // sorry c++17 only
      iz = std::min(std::max(iz, INT8_MIN), INT8_MAX);
      izt[i] = iz - INT8_MIN;
      assert(iz - INT8_MIN >= 0);
      assert(iz - INT8_MIN < 256);
      hist.count(izt[i]);
      iv[i] = i;
      nn[i] = 0;
    }
    __syncthreads();
    if (threadIdx.x < 32)
      hws[threadIdx.x] = 0;  // used by prefix scan...
    __syncthreads();
    hist.finalize(hws);
    __syncthreads();
    assert(hist.size() == nt);
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      hist.fill(izt[i], uint16_t(i));
    }
    __syncthreads();

    // count neighbours
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (ezt2[i] > er2mx)
        continue;
      auto loop = [&](uint32_t j) {
        if (i == j)
          return;
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > eps)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;
        nn[i]++;
      };

      cms::cuda::forEachInBins(hist, izt[i], 1, loop);
    }

    __syncthreads();

    // find closest above me .... (we ignore the possibility of two j at same distance from i)
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      float mdist = eps;
      auto loop = [&](uint32_t j) {
        if (nn[j] < nn[i])
          return;
        if (nn[j] == nn[i] && zt[j] >= zt[i])
          return;  // if equal use natural order...
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > mdist)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;  // (break natural order???)
        mdist = dist;
        iv[i] = j;  // assign to cluster (better be unique??)
      };
      cms::cuda::forEachInBins(hist, izt[i], 1, loop);
    }

    __syncthreads();

#ifdef GPU_DEBUG
    //  mini verification
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] != int(i))
        assert(iv[iv[i]] != int(i));
    }
    __syncthreads();
#endif

    // consolidate graph (percolate index of seed)
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      auto m = iv[i];
      while (m != iv[m])
        m = iv[m];
      iv[i] = m;
    }

#ifdef GPU_DEBUG
    __syncthreads();
    //  mini verification
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] != int(i))
        assert(iv[iv[i]] != int(i));
    }
#endif

#ifdef GPU_DEBUG
    // and verify that we did not spit any cluster...
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      auto minJ = i;
      auto mdist = eps;
      auto loop = [&](uint32_t j) {
        if (nn[j] < nn[i])
          return;
        if (nn[j] == nn[i] && zt[j] >= zt[i])
          return;  // if equal use natural order...
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > mdist)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;
        mdist = dist;
        minJ = j;
      };
      cms::cuda::forEachInBins(hist, izt[i], 1, loop);
      // should belong to the same cluster...
      assert(iv[i] == iv[minJ]);
      assert(nn[i] <= nn[iv[i]]);
    }
    __syncthreads();
#endif

    __shared__ unsigned int foundClusters;
    foundClusters = 0;
    __syncthreads();

    // find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
    // mark these tracks with a negative id.
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] == int(i)) {
        if (nn[i] >= minT) {
          auto old = atomicInc(&foundClusters, 0xffffffff);
          iv[i] = -(old + 1);
        } else {  // noise
          iv[i] = -9998;
        }
      }
    }
    __syncthreads();

    assert(foundClusters < zVertex::utilities::MAXVTX);

    // propagate the negative id to all the tracks in the cluster.
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] >= 0) {
        // mark each track in a cluster with the same id as the first one
        iv[i] = iv[iv[i]];
      }
    }
    __syncthreads();

    // adjust the cluster id to be a positive value starting from 0
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      iv[i] = -iv[i] - 1;
    }

    nvIntermediate = nvFinal = foundClusters;

    if (verbose && 0 == threadIdx.x)
      printf("found %d proto vertices\n", foundClusters);
  }

  __global__ void clusterTracksByDensityKernel(VtxSoAView pdata,
                                               WsSoAView pws,
                                               int minT,      // min number of neighbours to be "seed"
                                               float eps,     // max absolute distance to cluster
                                               float errmax,  // max error to be "seed"
                                               float chi2max  // max normalized distance to cluster
  ) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_plugins_gpuClusterTracksByDensity_h
