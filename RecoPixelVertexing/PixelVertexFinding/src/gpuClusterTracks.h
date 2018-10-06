#ifndef RecoPixelVertexing_PixelVertexFinding_clusterTracks_H
#define RecoPixelVertexing_PixelVertexFinding_clusterTracks_H

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __global__
  void sortByPt2(int nt,
                 OnGPU * pdata
                )  {
    auto & __restrict__ data = *pdata;
    float const * __restrict__ ptt2 = data.ptt2;
    uint32_t const & nv = *data.nv;

    int32_t const * __restrict__ iv = data.iv;
    float * __restrict__ ptv2 = data.ptv2;
    uint16_t * __restrict__ sortInd = data.sortInd;

    if (nv<1) return;

    // can be done asynchronoisly at the end of previous event
    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
      ptv2[i]=0;
    }
    __syncthreads();


    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i]>9990) continue;
      atomicAdd(&ptv2[iv[i]], ptt2[i]);
    }
    __syncthreads();

    if (1==nv) {
      if (threadIdx.x==0) sortInd[0]=0;
      return;
    }
    __shared__ uint16_t ws[1024];
    radixSort(ptv2,sortInd,ws,nv);
    
    assert(ptv2[sortInd[nv-1]]>=ptv2[sortInd[nv-2]]);
    assert(ptv2[sortInd[1]]>=ptv2[sortInd[0]]);
  }

  
  // this algo does not really scale as it works in a single block...
  // enough for <10K tracks we have
  __global__ 
  void clusterTracks(int nt,
		     OnGPU * pdata,
		     int minT,  // min number of neighbours to be "core"
		     float eps, // max absolute distance to cluster
		     float errmax, // max error to be "seed"
		     float chi2max   // max normalized distance to cluster
		     )  {

    constexpr bool verbose = false; // in principle the compiler should optmize out if false


    if(verbose && 0==threadIdx.x) printf("params %d %f\n",minT,eps);
    
    auto er2mx = errmax*errmax;
    
    auto & __restrict__ data = *pdata;
    float const * __restrict__ zt = data.zt;
    float const * __restrict__ ezt2 = data.ezt2;
    float * __restrict__ zv = data.zv;
    float * __restrict__ wv = data.wv;
    float * __restrict__ chi2 = data.chi2;
    uint32_t & nv = *data.nv;
    
    uint8_t  * __restrict__ izt = data.izt;
    int32_t * __restrict__ nn = data.nn;
    int32_t * __restrict__ iv = data.iv;
    
    assert(pdata);
    assert(zt);
    
    using Hist=HistoContainer<uint8_t,256,16000,8,uint16_t>;
    constexpr auto wss = Hist::totbins();
    __shared__ Hist hist;
    __shared__ typename Hist::Counter ws[wss];
    for (auto j=threadIdx.x; j<Hist::totbins(); j+=blockDim.x) { hist.off[j]=0; ws[j]=0;}
    __syncthreads();
    
    if(verbose && 0==threadIdx.x) printf("booked hist with %d bins, size %d for %d tracks\n",hist.nbins(),hist.capacity(),nt);
    
    assert(nt<=hist.capacity());
    
    // fill hist  (bin shall be wider than "eps")
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      assert(i<OnGPU::MAXTRACKS);
      int iz =  int(zt[i]*10.); // valid if eps<=0.1
      // iz = std::clamp(iz, INT8_MIN, INT8_MAX);  // sorry c++17 only
      iz = std::min(std::max(iz, INT8_MIN),INT8_MAX);
      izt[i]=iz-INT8_MIN;
      assert(iz-INT8_MIN >= 0);
      assert(iz-INT8_MIN < 256);
      hist.count(izt[i]);
      iv[i]=i;
      nn[i]=0;
    }
    __syncthreads();
    hist.finalize(ws);
    __syncthreads();
    assert(hist.size()==nt);
    if (threadIdx.x<32) ws[threadIdx.x]=0;  // used by prefix scan...
    __syncthreads();
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      hist.fill(izt[i],uint16_t(i),ws);
    }
    __syncthreads();    


    // count neighbours
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (ezt2[i]>er2mx) continue;
      auto loop = [&](int j) {
        if (i==j) return;
        auto dist = std::abs(zt[i]-zt[j]);
        if (dist>eps) return;
        if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return;
        nn[i]++;
      };

      forEachInBins(hist,izt[i],1,loop);
    }
      
    __syncthreads();
    
    // cluster seeds only
    bool more = true;
    while (__syncthreads_or(more)) {
      more=false;
      for (int  k = threadIdx.x; k < hist.size(); k += blockDim.x) {
        auto p = hist.begin()+k;
        auto i = (*p);
        auto be = std::min(Hist::bin(izt[i])+1,int(hist.nbins()-1));
	if (nn[i]<minT) continue; // DBSCAN core rule
	auto loop = [&](int j) {
	  assert (i!=j);
	  if (nn[j]<minT) return;  // DBSCAN core rule
          auto dist = std::abs(zt[i]-zt[j]);
          if (dist>eps) return;
	  if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return;
	  auto old = atomicMin(&iv[j], iv[i]);
	  if (old != iv[i]) {
	    // end the loop only if no changes were applied
	    more = true;
	  }
	  atomicMin(&iv[i], old);
	};
        ++p;
        for (;p<hist.end(be);++p) loop(*p);
      } // for i
    } // while
    
    
    
    // collect edges (assign to closest cluster of closest point??? here to closest point)
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      //    if (nn[i]==0 || nn[i]>=minT) continue;    // DBSCAN edge rule
      if (nn[i]>=minT) continue;    // DBSCAN edge rule
      float mdist=eps;
      auto loop = [&](int j) {
	if (nn[j]<minT) return;  // DBSCAN core rule
	auto dist = std::abs(zt[i]-zt[j]);
	if (dist>mdist) return;
	if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return; // needed?
	mdist=dist;
	iv[i] = iv[j]; // assign to cluster (better be unique??)
      };
      forEachInBins(hist,izt[i],1,loop);
    }
    
    
    __shared__ unsigned int foundClusters;
    foundClusters = 0;
    __syncthreads();
    
    // find the number of different clusters, identified by a tracks with clus[i] == i;
    // mark these tracks with a negative id.
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] == i) {
	if  (nn[i]>=minT) {
	  auto old = atomicInc(&foundClusters, 0xffffffff);
	  iv[i] = -(old + 1);
	  zv[old]=0;
	  wv[old]=0;
	  chi2[old]=0;
	} else { // noise
	  iv[i] = -9998;
	}
      }
    }
    __syncthreads();

    assert(foundClusters<OnGPU::MAXVTX);
    
    // propagate the negative id to all the tracks in the cluster.
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] >= 0) {
	// mark each track in a cluster with the same id as the first one
	iv[i] = iv[iv[i]];
      }
    }
    __syncthreads();
    
    // adjust the cluster id to be a positive value starting from 0
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      iv[i] = - iv[i] - 1;
    }
    
    // only for test
    __shared__ int noise;
   if(verbose && 0==threadIdx.x) noise = 0;
    
    __syncthreads();
    
    // compute cluster location
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i]>9990) {
	if (verbose) atomicAdd(&noise, 1);
	continue;
      }
      assert(iv[i]>=0);
      assert(iv[i]<foundClusters);
      // if (nn[i]<minT) continue;  //  ONLY?? DBSCAN core rule
      auto w = 1.f/ezt2[i];
      atomicAdd(&zv[iv[i]],zt[i]*w);
      atomicAdd(&wv[iv[i]],w); 
    }
    
    __syncthreads();
    // reuse nn 
    for (int i = threadIdx.x; i < foundClusters; i += blockDim.x) {
      assert(wv[i]>0.f);
      zv[i]/=wv[i];
      nn[i]=-1;  // ndof
    }
    __syncthreads();
 
    
    // compute chi2
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i]>9990) continue;
    
      auto c2 = zv[iv[i]]-zt[i]; c2 *=c2/ezt2[i];
      // remove outliers ???? if (c2> cut) {iv[i] = 9999; continue;}????
      atomicAdd(&chi2[iv[i]],c2);
      atomicAdd(&nn[iv[i]],1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < foundClusters; i += blockDim.x) if(nn[i]>0) wv[i] *= float(nn[i])/chi2[i];
    
    if(verbose && 0==threadIdx.x) printf("found %d proto clusters ",foundClusters);
    if(verbose && 0==threadIdx.x) printf("and %d noise\n",noise);
    
    nv = foundClusters;
  }

}
#endif
