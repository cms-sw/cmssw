#ifndef RecoPixelVertexing_PixelVertexFinding_sortByPt2_H
#define RecoPixelVertexing_PixelVertexFinding_sortByPt2_H


#include<cstdint>
#include<cmath>
#include <algorithm>
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"


#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __global__
  void sortByPt2(
                 OnGPU * pdata
                )  {
    auto & __restrict__ data = *pdata;
    auto nt = *data.ntrks;
    float const * __restrict__ ptt2 = data.ptt2;
    uint32_t const & nvFinal = *data.nvFinal;

    int32_t const * __restrict__ iv = data.iv;
    float * __restrict__ ptv2 = data.ptv2;
    uint16_t * __restrict__ sortInd = data.sortInd;

    if (nvFinal<1) return;

    // can be done asynchronoisly at the end of previous event
    for (int i = threadIdx.x; i < nvFinal; i += blockDim.x) {
      ptv2[i]=0;
    }
    __syncthreads();


    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i]>9990) continue;
      atomicAdd(&ptv2[iv[i]], ptt2[i]);
    }
    __syncthreads();

    if (1==nvFinal) {
      if (threadIdx.x==0) sortInd[0]=0;
      return;
    }
    __shared__ uint16_t ws[1024];
    radixSort(ptv2,sortInd,ws,nvFinal);

    assert(ptv2[sortInd[nvFinal-1]]>=ptv2[sortInd[nvFinal-2]]);
    assert(ptv2[sortInd[1]]>=ptv2[sortInd[0]]);
  }

}

#endif
