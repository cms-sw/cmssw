#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "gpuClusteringConstants.h"

namespace gpuClustering {

  __global__ void countModules(uint16_t const * id,
                               uint32_t * moduleStart,
                               int32_t * clus,
                               int numElements)
  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numElements)
      return;
    clus[i] = i;
    if (InvId == id[i])
      return;
    auto j = i - 1;
    while (j >= 0 and id[j] == InvId)
      --j;
    if (j < 0 or id[j] != id[i]) {
      // boundary...
      auto loc = atomicInc(moduleStart, MaxNumModules);
      moduleStart[loc + 1] = i;
    }
  }

  __global__ void findClus(uint16_t const * id,
                           uint16_t const * x,
                           uint16_t const * y,
                           uint16_t const * adc,
                           uint32_t const * moduleStart,
                           uint32_t * clusInModule, uint32_t * moduleId,
                           int32_t * clus,
                           int numElements)
  {
    __shared__ bool go;
    __shared__ int nclus;
    __shared__ int msize;

    if (blockIdx.x >= moduleStart[0])
      return;

    auto first = moduleStart[1 + blockIdx.x];
    auto me = id[first];

    assert(me < MaxNumModules);

#ifdef GPU_DEBUG
    if (me%100 == 1)
      if (threadIdx.x == 0)
        printf("start clusterizer for module %d in block %d\n", me, blockIdx.x);
#endif

    first += threadIdx.x;
    if (first>= numElements)
      return;

    go = true;
    nclus = 0;

    msize = numElements;
    __syncthreads();

    for (int i = first; i < numElements; i += blockDim.x) {
      if (id[i] == InvId)               // not valid
        continue;
      if (id[i] != me) {                // end of module
        atomicMin(&msize, i);
        break;
      }
    }
    __syncthreads();

    assert(msize<= numElements);
    if (first>= msize)
      return;

    int jmax[10];
    auto niter = (msize - first) / blockDim.x;
    assert(niter < 10);
    for (int k = 0; k < niter + 1; ++k)
      jmax[k] = msize;

    while (go) {
      __syncthreads();
      go = false;

      __syncthreads();
     int k = -1;
      for (int i = first; i < msize; i += blockDim.x) {
        ++k;
        if (id[i] == InvId)             // not valid
          continue;
        assert(id[i] == me); // break;  // end of module
        auto js = i + 1;
        auto jm = jmax[k];
        jmax[k] = i + 1;
        for (int j = js; j < jm; ++j) {
          if (id[j] == InvId)           // not valid
            continue;
          if (std::abs(int(x[j]) - int(x[i])) > 1 |
              std::abs(int(y[j]) - int(y[i])) > 1)
            continue;
          auto old = atomicMin(&clus[j], clus[i]);
          if (old != clus[i]) go = true;
          atomicMin(&clus[i], old);
          jmax[k] = j + 1;
        }
      }
      assert (k<= niter);
      __syncthreads();
    }

    nclus = 0;
    __syncthreads();
    for (int i = first; i < numElements; i += blockDim.x) {
      if (id[i] == InvId)               // not valid
        continue;
      if (id[i] != me)                  // end of module
        break;
      if (clus[i] == i) {
        auto old = atomicAdd(&nclus, 1);
        clus[i] = -(old + 1);
      }
    }

    __syncthreads();
    for (int i = first; i < numElements; i += blockDim.x) {
      if (id[i] == InvId)               // not valid
        continue;
      if (id[i] != me)                  // end of module
        break;
      if (clus[i]>= 0) clus[i] = clus[clus[i]];
    }

    __syncthreads();
    for (int i = first; i < numElements; i += blockDim.x) {
      if (id[i] == InvId) {             // not valid
        clus[i] = -9999;
        continue;
      }
      if (id[i] != me)                  // end of module
        break;
      clus[i] = - clus[i] - 1;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      clusInModule[me] = nclus;
      moduleId[blockIdx.x] = me;
    }

#ifdef GPU_DEBUG
    if (me % 100 == 1)
      if (threadIdx.x == 0)
        printf("%d clusters in module %d\n", nclus, me);
#endif
  }

} // namespace gpuClustering

#endif // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
