#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "gpuClusteringConstants.h"

namespace gpuClustering {

  __global__ void countModules(uint16_t const * id,
                               uint32_t * moduleStart,
                               int32_t * clusterId,
                               int numElements)
  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numElements)
      return;
    clusterId[i] = i;
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

  __global__ void findClus(uint16_t const * id,             // module id of each pixel
                           uint16_t const * x,              // local coordinates of each pixel
                           uint16_t const * y,              //
                           uint32_t const * moduleStart,    // index of the first pixel of each module
                           uint32_t * nClustersInModule,    // output: number of clusters found in each module
                           uint32_t * moduleId,             // output: module id of each module
                           int32_t *  clusterId,            // output: cluster id of each pixel
                           int numElements)
  {
    if (blockIdx.x >= moduleStart[0])
      return;

    auto firstPixel = moduleStart[1 + blockIdx.x];
    auto thisModuleId = id[firstPixel];
    assert(thisModuleId < MaxNumModules);

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (threadIdx.x == 0)
        printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx.x);
#endif

    auto first = firstPixel + threadIdx.x;

    // find the index of the first pixel not belonging to this module (or invalid)
    __shared__ int msize;
    msize = numElements;
    __syncthreads();

    // skip threads not associated to an existing pixel
    bool active = (first < numElements);
    if (active) {
      for (int i = first; i < numElements; i += blockDim.x) {
        if (id[i] == InvId)                 // skip invalid pixels
          continue;
        if (id[i] != thisModuleId) {        // find the first pixel in a different module
          atomicMin(&msize, i);
          break;
        }
      }
    }

    __syncthreads();
    assert((msize == numElements) or ((msize < numElements) and (id[msize] != thisModuleId)));

    // skip threads not assocoated to pixels in this module
    active = (first < msize);

    // assume that we can cover the whole module with up to 10 blockDim.x-wide iterations
    constexpr int maxiter = 10;
    if (active) {
      assert(((msize - first) / blockDim.x) <= maxiter);
    }
    int jmax[maxiter];
    for (int k = 0; k < maxiter; ++k)
      jmax[k] = msize;

    __syncthreads();
    // for each pixel, look at all the pixels until the end of the module;
    // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
    // after the loop, all the pixel in each cluster should have the id equeal to the lowest
    // pixel in the cluster ( clus[i] == i ).
    bool done = false;
    while (not __syncthreads_and(done)) {
      done = true;
      if (active) {
        for (int i = first, k = 0; i < msize; i += blockDim.x, ++k) {
          if (id[i] == InvId)               // skip invalid pixels
            continue;
          assert(id[i] == thisModuleId);    // same module
          auto js = i + 1;
          auto jm = jmax[k];
          jmax[k] = i + 1;
          for (int j = js; j < jm; ++j) {
            if (id[j] == InvId)             // skip invalid pixels
              continue;
            if (std::abs(int(x[j]) - int(x[i])) > 1 or
                std::abs(int(y[j]) - int(y[i])) > 1)
              continue;
            auto old = atomicMin(&clusterId[j], clusterId[i]);
            if (old != clusterId[i]) {
              // end the loop only if no changes were applied
              done = false;
            }
            atomicMin(&clusterId[i], old);
            // update the loop boundary for the next iteration
            jmax[k] = j + 1;
          }
        }
      }
    }

    __shared__ int foundClusters;
    foundClusters = 0;
    __syncthreads();

    // find the number of different clusters, identified by a pixels with clus[i] == i;
    // mark these pixels with a negative id.
    if (active) {
      for (int i = first; i < numElements; i += blockDim.x) {
        if (id[i] == InvId)                 // skip invalid pixels
          continue;
        if (id[i] != thisModuleId)          // stop once in a different module
          break;
        if (clusterId[i] == i) {
          auto old = atomicAdd(&foundClusters, 1);
          clusterId[i] = -(old + 1);
        }
      }
    }
    __syncthreads();

    // propagate the negative id to all the pixels in the cluster.
    if (active) {
      for (int i = first; i < numElements; i += blockDim.x) {
        if (id[i] == InvId)                 // skip invalid pixels
          continue;
        if (id[i] != thisModuleId)          // stop once in a different module
          break;
        if (clusterId[i] >= 0) {
          // mark each pixel in a cluster with the same id as the first one
          clusterId[i] = clusterId[clusterId[i]];
        }
      }
    }
    __syncthreads();

    // adjust the cluster id to be a positive value starting from 0
    if (active) {
      for (int i = first; i < numElements; i += blockDim.x) {
        if (id[i] == InvId) {               // skip invalid pixels
          clusterId[i] = -9999;
          continue;
        }
        if (id[i] != thisModuleId)          // stop once in a different module
          break;
        clusterId[i] = - clusterId[i] - 1;
      }
    }
    __syncthreads();

    if (active) {
      if (threadIdx.x == 0) {
        nClustersInModule[thisModuleId] = foundClusters;
        moduleId[blockIdx.x] = thisModuleId;
      }
#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
        if (threadIdx.x == 0)
          printf("%d clusters in module %d\n", foundClusters, thisModuleId);
#endif
    }
  }

} // namespace gpuClustering

#endif // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
