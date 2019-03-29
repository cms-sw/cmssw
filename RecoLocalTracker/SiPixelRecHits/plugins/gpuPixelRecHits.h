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


  __global__ void getHits(pixelCPEforGPU::ParamsOnGPU const * __restrict__  cpeParams,
                          BeamSpotCUDA::Data const * __restrict__  bs,
                          uint16_t const * __restrict__  id,
			  uint16_t const * __restrict__  x,
			  uint16_t const * __restrict__  y,
			  uint16_t const * __restrict__  adc,
			  uint32_t const * __restrict__  digiModuleStart,
			  uint32_t const * __restrict__  clusInModule,
			  uint32_t const * __restrict__  moduleId,
			  int32_t  const * __restrict__  clus,
			  int numElements,
			  uint32_t const * __restrict__  hitsModuleStart,
                          TrackingRecHit2DSOAView * phits
                         )
{

    auto & hits = *phits;

    // to be moved in common namespace...
    constexpr uint16_t InvId=9999; // must be > MaxNumModules
    constexpr uint32_t MaxHitsInModule = pixelCPEforGPU::MaxHitsInModule;

    using ClusParams = pixelCPEforGPU::ClusParams;


    // as usual one block per module
    __shared__ ClusParams clusParams;

    auto first = digiModuleStart[1 + blockIdx.x];
    auto me = moduleId[blockIdx.x];
    auto nclus = clusInModule[me];

    if (0==nclus) return;

#ifdef GPU_DEBUG
    if (threadIdx.x==0) {
      auto k=first;
      while (id[k]==InvId) ++k;
      assert(id[k]==me);
    }
#endif

#ifdef GPU_DEBUG
    if (me%100==1)
      if (threadIdx.x==0) printf("hitbuilder: %d clusters in module %d. will write at %d\n", nclus, me, hitsModuleStart[me]);
#endif

    assert(blockDim.x >= MaxHitsInModule);

    if (threadIdx.x==0 && nclus > MaxHitsInModule) { 
      printf("WARNING: too many clusters %d in Module %d. Only first %d processed\n", nclus,me,MaxHitsInModule);
      // zero charge: do not bother to do it in parallel
      for (auto d=MaxHitsInModule; d<nclus; ++d) { hits.charge(d)=0; hits.detectorIndex(d)=InvId;}
    }
    nclus = std::min(nclus, MaxHitsInModule);

    auto ic = threadIdx.x;

    if (ic < nclus) {
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
      if (id[i] == InvId) continue;     // not valid
      if (id[i] != me) break;           // end of module
      if (clus[i] >= nclus) continue;
      atomicMin(&clusParams.minRow[clus[i]], x[i]);
      atomicMax(&clusParams.maxRow[clus[i]], x[i]);
      atomicMin(&clusParams.minCol[clus[i]], y[i]);
      atomicMax(&clusParams.maxCol[clus[i]], y[i]);
    }

    __syncthreads();

    for (int i = first; i < numElements; i += blockDim.x) {
      if (id[i] == InvId) continue;     // not valid
      if (id[i] != me) break;           // end of module
      if (clus[i] >= nclus) continue;
      atomicAdd(&clusParams.charge[clus[i]], adc[i]);
      if (clusParams.minRow[clus[i]]==x[i]) atomicAdd(&clusParams.Q_f_X[clus[i]], adc[i]);
      if (clusParams.maxRow[clus[i]]==x[i]) atomicAdd(&clusParams.Q_l_X[clus[i]], adc[i]);
      if (clusParams.minCol[clus[i]]==y[i]) atomicAdd(&clusParams.Q_f_Y[clus[i]], adc[i]);
      if (clusParams.maxCol[clus[i]]==y[i]) atomicAdd(&clusParams.Q_l_Y[clus[i]], adc[i]);
    }

    __syncthreads();

    // next one cluster per thread...

    if (ic >= nclus) return;

    first = hitsModuleStart[me];
    auto h = first+ic;  // output index in global memory

    if (h >= TrackingRecHit2DSOAView::maxHits()) return; // overflow...

    pixelCPEforGPU::position(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);
    pixelCPEforGPU::errorFromDB(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);


    // store it

    hits.charge(h) = clusParams.charge[ic];

    hits.detectorIndex(h) = me;

    float xl,yl;
    hits.xLocal(h) = xl = clusParams.xpos[ic];   
    hits.yLocal(h) = yl = clusParams.ypos[ic]; 

    hits.clusterSizeX(h) = clusParams.xsize[ic];
    hits.clusterSizeY(h) = clusParams.ysize[ic];


    hits.xerrLocal(h) = clusParams.xerr[ic]*clusParams.xerr[ic];
    hits.yerrLocal(h) = clusParams.yerr[ic]*clusParams.yerr[ic];

    // keep it local for computations
    float xg,yg,zg;  
    // to global and compute phi... 
    cpeParams->detParams(me).frame.toGlobal(xl,yl, xg,yg,zg);
    // here correct for the beamspot...
    xg-=bs->x;
    yg-=bs->y;
    zg-=bs->z;

    hits.xGlobal(h) = xg;
    hits.yGlobal(h) = yg;
    hits.zGlobal(h) = zg;

    hits.rGlobal(h) = std::sqrt(xg*xg+yg*yg);
    hits.iphi(h) = unsafe_atan2s<7>(yg,xg);
    
  }

}

#endif // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
