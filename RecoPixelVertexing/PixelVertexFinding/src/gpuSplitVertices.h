#ifndef RecoPixelVertexing_PixelVertexFinding_splitVertices_H
#define RecoPixelVertexing_PixelVertexFinding_splitVertices_H

#include<cstdint>
#include<cmath>
#include <algorithm>
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"


#include "gpuVertexFinder.h"

namespace gpuVertexFinder {


  __global__
  void splitVertices(
                   OnGPU * pdata,
                   float maxChi2
                  )  {

    constexpr bool verbose = false; // in principle the compiler should optmize out if false


    auto & __restrict__ data = *pdata;
    auto nt = *data.ntrks;
    float const * __restrict__ zt = data.zt;
    float const * __restrict__ ezt2 = data.ezt2;
    float * __restrict__ zv = data.zv;
    float * __restrict__ wv = data.wv;
    float const * __restrict__ chi2 = data.chi2;
    uint32_t & nvFinal  = *data.nvFinal;

    int32_t const * __restrict__ nn = data.nn;
    int32_t * __restrict__ iv = data.iv;

    assert(pdata);
    assert(zt);

    // one vertex per block
    auto kv = blockIdx.x;
    
    if (kv>= nvFinal) return;
    if (nn[kv]<4) return;
    if (chi2[kv]<maxChi2*float(nn[kv])) return;
    
    assert(nn[kv]<1023);
    __shared__ uint32_t it[1024];   // track index
    __shared__ float zz[1024];      // z pos
    __shared__ uint8_t newV[1024];   // 0 or 1
    __shared__ float ww[1024];      // z weight
    
    __shared__ uint32_t nq;  // number of track for this vertex
    nq=0;
    __syncthreads();

    // copy to local
    for (auto k = threadIdx.x; k<nt; k+=blockDim.x) {
      if (iv[k]==kv) {
        auto old = atomicInc(&nq,1024);
        zz[old] = zt[k]-zv[kv];
        newV[old] = zz[old]<0 ? 0 : 1;
        ww[old] = 1.f/ezt2[k];
        it[old] = k;
      }
    }
    
    __shared__ float znew[2], wnew[2];  // the new vertices
    
    __syncthreads();
    assert(nq==nn[kv]+1);
    

    int  maxiter=20;
    // kt-min....
    bool more = true;
    while(__syncthreads_or(more) ) {
      more = false;
      if(0==threadIdx.x) {
        znew[0]=0; znew[1]=0;
        wnew[0]=0; wnew[1]=0;
      }
      __syncthreads();
      for (auto k = threadIdx.x; k<nq; k+=blockDim.x) {
        auto i = newV[k];
        atomicAdd(&znew[i],zz[k]*ww[k]);
        atomicAdd(&wnew[i],ww[k]);
      }
      __syncthreads();
      if(0==threadIdx.x) {
        znew[0]/=wnew[0];
        znew[1]/=wnew[1];
      }
      __syncthreads();
      for (auto k = threadIdx.x; k<nq; k+=blockDim.x) {
        auto d0 = fabs(zz[k]-znew[0]);
        auto d1 = fabs(zz[k]-znew[1]);
        auto newer = d0<d1 ? 0 : 1;
        more |= newer != newV[k];
        newV[k] = newer;
      }
      --maxiter;
      if (maxiter<=0) more=false;
    }
    
    // quality cut
    auto dist2 = (znew[0]-znew[1])*(znew[0]-znew[1]);

    auto chi2Dist = dist2/(1.f/wnew[0]+1.f/wnew[1]);

    if(verbose && 0==threadIdx.x) printf("inter %d %f %f\n",20-maxiter,chi2Dist, dist2*wv[kv]);
    
    if (chi2Dist<4) return;
    
    // get a new global vertex
    __shared__ uint32_t igv;
    if (0==threadIdx.x) igv = atomicInc(data.nvIntermediate,1024);
    __syncthreads();
    for (auto k = threadIdx.x; k<nq; k+=blockDim.x) {
      if(1==newV[k]) iv[it[k]]=igv;
    }

  }

}

#endif
