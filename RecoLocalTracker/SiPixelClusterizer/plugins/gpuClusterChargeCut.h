#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h

#include <cstdint>
#include <cstdio>

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/prefixScan.h"

#include "gpuClusteringConstants.h"

namespace gpuClustering {

  __global__ void clusterChargeCut(
      uint16_t* __restrict__ id,                 // module id of each pixel (modified if bad cluster)
      uint16_t const* __restrict__ adc,          //  charge of each pixel
      uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
      uint32_t* __restrict__ nClustersInModule,  // modified: number of clusters found in each module
      uint32_t const* __restrict__ moduleId,     // module id of each module
      int32_t* __restrict__ clusterId,           // modified: cluster id of each pixel
      uint32_t numElements) {
    if (blockIdx.x >= moduleStart[0])
      return;

    auto firstPixel = moduleStart[1 + blockIdx.x];
    auto thisModuleId = id[firstPixel];
    assert(thisModuleId < MaxNumModules);
    assert(thisModuleId == moduleId[blockIdx.x]);

    auto nclus = nClustersInModule[thisModuleId];
    if (nclus == 0)
      return;

    if (threadIdx.x == 0 && nclus > MaxNumClustersPerModules)
      printf("Warning too many clusters in module %d in block %d: %d > %d\n",
             thisModuleId,
             blockIdx.x,
             nclus,
             MaxNumClustersPerModules);

    auto first = firstPixel + threadIdx.x;

    if (nclus > MaxNumClustersPerModules) {
      // remove excess  FIXME find a way to cut charge first....
      for (auto i = first; i < numElements; i += blockDim.x) {
        if (id[i] == InvId)
          continue;  // not valid
        if (id[i] != thisModuleId)
          break;  // end of module
        if (clusterId[i] >= MaxNumClustersPerModules) {
          id[i] = InvId;
          clusterId[i] = InvId;
        }
      }
      nclus = MaxNumClustersPerModules;
    }

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (threadIdx.x == 0)
        printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx.x);
#endif

    __shared__ int32_t charge[MaxNumClustersPerModules];
    __shared__ uint8_t ok[MaxNumClustersPerModules];
    __shared__ uint16_t newclusId[MaxNumClustersPerModules];

    assert(nclus <= MaxNumClustersPerModules);
    for (auto i = threadIdx.x; i < nclus; i += blockDim.x) {
      charge[i] = 0;
    }
    __syncthreads();

    for (auto i = first; i < numElements; i += blockDim.x) {
      if (id[i] == InvId)
        continue;  // not valid
      if (id[i] != thisModuleId)
        break;  // end of module
      atomicAdd(&charge[clusterId[i]], adc[i]);
    }
    __syncthreads();

    auto chargeCut = thisModuleId < 96 ? 2000 : 4000;  // move in constants (calib?)
    for (auto i = threadIdx.x; i < nclus; i += blockDim.x) {
      newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0;
    }

    __syncthreads();

    // renumber
    __shared__ uint16_t ws[32];
    cms::cuda::blockPrefixScan(newclusId, nclus, ws);

    assert(nclus >= newclusId[nclus - 1]);

    if (nclus == newclusId[nclus - 1])
      return;

    nClustersInModule[thisModuleId] = newclusId[nclus - 1];
    __syncthreads();

    // mark bad cluster again
    for (auto i = threadIdx.x; i < nclus; i += blockDim.x) {
      if (0 == ok[i])
        newclusId[i] = InvId + 1;
    }
    __syncthreads();

    // reassign id
    for (auto i = first; i < numElements; i += blockDim.x) {
      if (id[i] == InvId)
        continue;  // not valid
      if (id[i] != thisModuleId)
        break;  // end of module
      clusterId[i] = newclusId[clusterId[i]] - 1;
      if (clusterId[i] == InvId)
        id[i] = InvId;
    }

    //done
  }

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
