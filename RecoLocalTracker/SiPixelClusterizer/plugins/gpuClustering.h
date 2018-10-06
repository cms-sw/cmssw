#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

// #define CLUS_LIMIT_LOOP

#include <cstdint>
#include <cstdio>

#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "gpuClusteringConstants.h"

namespace gpuClustering {

  __global__ void countModules(uint16_t const * __restrict__ id,
                               uint32_t * __restrict__ moduleStart,
                               int32_t * __restrict__ clusterId,
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

  __global__
//  __launch_bounds__(256,4)
  void findClus(uint16_t const * __restrict__ id,             // module id of each pixel
                           uint16_t const * __restrict__ x,              // local coordinates of each pixel
                           uint16_t const * __restrict__ y,              //
                           uint32_t const * __restrict__ moduleStart,    // index of the first pixel of each module
                           uint32_t * __restrict__ nClustersInModule,    // output: number of clusters found in each module
                           uint32_t * __restrict__ moduleId,             // output: module id of each module
                           int32_t * __restrict__  clusterId,            // output: cluster id of each pixel
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
      for (int i = first; i < numElements; i += blockDim.x) {
        if (id[i] == InvId)                 // skip invalid pixels
          continue;
        if (id[i] != thisModuleId) {        // find the first pixel in a different module
          atomicMin(&msize, i);
          break;
        }
      }

   //init hist  (ymax=416 < 512 : 9bits)
   constexpr uint32_t maxPixInModule = 4000;
   constexpr auto  nbins = phase1PixelTopology::numColsInModule + 2;   //2+2;
   using Hist = HistoContainer<uint16_t,nbins,maxPixInModule,9,uint16_t>;
   constexpr auto wss = Hist::totbins();
    __shared__ Hist hist;
    __shared__ typename Hist::Counter ws[wss];
    for (auto j=threadIdx.x; j<Hist::totbins(); j+=blockDim.x) { hist.off[j]=0; ws[j]=0;}
    __syncthreads();

    assert((msize == numElements) or ((msize < numElements) and (id[msize] != thisModuleId)));
    assert(msize-firstPixel<maxPixInModule);  
 

#ifdef GPU_DEBUG
    __shared__ uint32_t totGood;
    totGood=0;
    __syncthreads();
#endif

    // fill histo
      for (int i = first; i < msize; i += blockDim.x) {
        if (id[i] == InvId)                 // skip invalid pixels
          continue;
        hist.count(y[i]);
#ifdef GPU_DEBUG
        atomicAdd(&totGood,1);
#endif
      }
    __syncthreads();
    hist.finalize(ws);
    __syncthreads();
    if (threadIdx.x<32) ws[threadIdx.x]=0;  // used by prefix scan...
    __syncthreads();
#ifdef GPU_DEBUG
    assert(hist.size()==totGood);
    if (thisModuleId % 100 == 1)
      if (threadIdx.x == 0)
        printf("histo size %d\n",hist.size());
#endif
      for (int i = first; i < msize; i += blockDim.x) {
        if (id[i] == InvId)                 // skip invalid pixels
          continue;
        hist.fill(y[i],i-firstPixel,ws);
      }

#ifdef CLUS_LIMIT_LOOP
    // assume that we can cover the whole module with up to 10 blockDim.x-wide iterations
    constexpr int maxiter = 10;
    if (threadIdx.x==0) {
      assert((hist.size()/ blockDim.x) <= maxiter);
    }
    uint16_t const * jmax[maxiter];
    for (int k = 0; k < maxiter; ++k)
      jmax[k] = hist.end();
#endif

#ifdef GPU_DEBUG
    __shared__ int nloops;
    nloops=0;
#endif


    __syncthreads();  // for hit filling!

    // for each pixel, look at all the pixels until the end of the module;
    // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
    // after the loop, all the pixel in each cluster should have the id equeal to the lowest
    // pixel in the cluster ( clus[i] == i ).
    bool more = true;
    while (__syncthreads_or(more)) {
      more = false;
        for (int j=threadIdx.x, k = 0; j<hist.size(); j+=blockDim.x, ++k) {
          auto p = hist.begin()+j;
          auto i = *p + firstPixel;
          assert (id[i] != InvId);
          assert(id[i] == thisModuleId);    // same module
#ifdef CLUS_LIMIT_LOOP
          auto jm = jmax[k];
          jmax[k] = p + 1;
#endif
          int be = Hist::bin(y[i]+1);
          auto e = hist.end(be);
#ifdef CLUS_LIMIT_LOOP
          e = std::min(e,jm);
#endif      
          // loop to columns
          auto loop = [&](uint16_t const * kk) {
            auto m = (*kk)+firstPixel;
#ifdef GPU_DEBUG
            assert(m!=i);
#endif
            if (std::abs(int(x[m]) - int(x[i])) > 1) return;
            // if (std::abs(int(y[m]) - int(y[i])) > 1) return; // binssize is 1
            auto old = atomicMin(&clusterId[m], clusterId[i]);
            if (old != clusterId[i]) {
              // end the loop only if no changes were applied
              more = true;
            }
            atomicMin(&clusterId[i], old);
#ifdef CLUS_LIMIT_LOOP
            // update the loop boundary for the next iteration
            jmax[k] = std::max(kk + 1,jmax[k]);
#endif
          };
          ++p;
          for (;p<e;++p) loop(p);
        } // pixel loop
#ifdef GPU_DEBUG
        if (threadIdx.x==0) ++nloops;
#endif
    }  // end while

#ifdef GPU_DEBUG
   if (thisModuleId % 100 == 1)
      if (threadIdx.x == 0)
        printf("# loops %d\n",nloops);
#endif

    __shared__ unsigned int foundClusters;
    foundClusters = 0;
    __syncthreads();

    // find the number of different clusters, identified by a pixels with clus[i] == i;
    // mark these pixels with a negative id.
      for (int i = first; i < msize; i += blockDim.x) {
        if (id[i] == InvId)                 // skip invalid pixels
          continue;
        if (clusterId[i] == i) {
          auto old = atomicInc(&foundClusters, 0xffffffff);
          clusterId[i] = -(old + 1);
        }
      }
    __syncthreads();

    // propagate the negative id to all the pixels in the cluster.
      for (int i = first; i < msize; i += blockDim.x) {
        if (id[i] == InvId)                 // skip invalid pixels
          continue;
        if (clusterId[i] >= 0) {
          // mark each pixel in a cluster with the same id as the first one
          clusterId[i] = clusterId[clusterId[i]];
        }
      }
    __syncthreads();

    // adjust the cluster id to be a positive value starting from 0
      for (int i = first; i < msize; i += blockDim.x) {
        if (id[i] == InvId) {               // skip invalid pixels
          clusterId[i] = -9999;
          continue;
        }
        clusterId[i] = - clusterId[i] - 1;
      }
    __syncthreads();

      if (threadIdx.x == 0) {
        nClustersInModule[thisModuleId] = foundClusters;
        moduleId[blockIdx.x] = thisModuleId;
#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
        if (threadIdx.x == 0)
          printf("%d clusters in module %d\n", foundClusters, thisModuleId);
#endif
    }
  }

} // namespace gpuClustering

#endif // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
