#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __device__ __forceinline__ void splitVertices(ZVertices* pdata, WorkSpace* pws, float maxChi2) {
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ zt = ws.zt;
    float const* __restrict__ ezt2 = ws.ezt2;
    float* __restrict__ zv = data.zv;
    float* __restrict__ wv = data.wv;
    float const* __restrict__ chi2 = data.chi2;
    uint32_t& nvFinal = data.nvFinal;

    int32_t const* __restrict__ nn = data.ndof;
    int32_t* __restrict__ iv = ws.iv;

    assert(pdata);
    assert(zt);

    // one vertex per block
    for (auto kv = blockIdx.x; kv < nvFinal; kv += gridDim.x) {
      if (nn[kv] < 4)
        continue;
      if (chi2[kv] < maxChi2 * float(nn[kv]))
        continue;

      constexpr int MAXTK = 512;
      assert(nn[kv] < MAXTK);
      if (nn[kv] >= MAXTK)
        continue;                      // too bad FIXME
      __shared__ uint32_t it[MAXTK];   // track index
      __shared__ float zz[MAXTK];      // z pos
      __shared__ uint8_t newV[MAXTK];  // 0 or 1
      __shared__ float ww[MAXTK];      // z weight

      __shared__ uint32_t nq;  // number of track for this vertex
      nq = 0;
      __syncthreads();

      // copy to local
      for (auto k = threadIdx.x; k < nt; k += blockDim.x) {
        if (iv[k] == int(kv)) {
          auto old = atomicInc(&nq, MAXTK);
          zz[old] = zt[k] - zv[kv];
          newV[old] = zz[old] < 0 ? 0 : 1;
          ww[old] = 1.f / ezt2[k];
          it[old] = k;
        }
      }

      __shared__ float znew[2], wnew[2];  // the new vertices

      __syncthreads();
      assert(int(nq) == nn[kv] + 1);

      int maxiter = 20;
      // kt-min....
      bool more = true;
      while (__syncthreads_or(more)) {
        more = false;
        if (0 == threadIdx.x) {
          znew[0] = 0;
          znew[1] = 0;
          wnew[0] = 0;
          wnew[1] = 0;
        }
        __syncthreads();
        for (auto k = threadIdx.x; k < nq; k += blockDim.x) {
          auto i = newV[k];
          atomicAdd(&znew[i], zz[k] * ww[k]);
          atomicAdd(&wnew[i], ww[k]);
        }
        __syncthreads();
        if (0 == threadIdx.x) {
          znew[0] /= wnew[0];
          znew[1] /= wnew[1];
        }
        __syncthreads();
        for (auto k = threadIdx.x; k < nq; k += blockDim.x) {
          auto d0 = fabs(zz[k] - znew[0]);
          auto d1 = fabs(zz[k] - znew[1]);
          auto newer = d0 < d1 ? 0 : 1;
          more |= newer != newV[k];
          newV[k] = newer;
        }
        --maxiter;
        if (maxiter <= 0)
          more = false;
      }

      // avoid empty vertices
      if (0 == wnew[0] || 0 == wnew[1])
        continue;

      // quality cut
      auto dist2 = (znew[0] - znew[1]) * (znew[0] - znew[1]);

      auto chi2Dist = dist2 / (1.f / wnew[0] + 1.f / wnew[1]);

      if (verbose && 0 == threadIdx.x)
        printf("inter %d %f %f\n", 20 - maxiter, chi2Dist, dist2 * wv[kv]);

      if (chi2Dist < 4)
        continue;

      // get a new global vertex
      __shared__ uint32_t igv;
      if (0 == threadIdx.x)
        igv = atomicAdd(&ws.nvIntermediate, 1);
      __syncthreads();
      for (auto k = threadIdx.x; k < nq; k += blockDim.x) {
        if (1 == newV[k])
          iv[it[k]] = igv;
      }

    }  // loop on vertices
  }

  __global__ void splitVerticesKernel(ZVertices* pdata, WorkSpace* pws, float maxChi2) {
    splitVertices(pdata, pws, maxChi2);
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
