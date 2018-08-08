#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"

#include "GPUCACell.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"

namespace gpuPixelDoublets {

  constexpr uint32_t MaxNumOfDoublets = 1024*1024*256;

  template<typename Hist>
  __device__
  void doubletsFromHisto(uint8_t const * layerPairs, uint32_t nPairs, GPUCACell * cells, uint32_t * nCells,
                         int16_t const * iphi, Hist const * hist, uint32_t const * offsets,
                         siPixelRecHitsHeterogeneousProduct::HitsOnGPU const & hh,
                         GPU::VecArray< unsigned int, 2048>  * isOuterHitOfCell,
                         int16_t const * phicuts, float const * minz, float const * maxz, float const * maxr) {

    auto layerSize = [=](uint8_t li) { return offsets[li+1]-offsets[li]; };

    // to be optimized later
    uint32_t innerLayerCumulativeSize[64];
    assert(nPairs<=64);
    innerLayerCumulativeSize[0] = layerSize(layerPairs[0]);
    for (uint32_t i=1; i<nPairs; ++i) {
       innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i-1] + layerSize(layerPairs[2*i]);
    }

    auto ntot = innerLayerCumulativeSize[nPairs-1];


    auto idx = blockIdx.x*blockDim.x + threadIdx.x;
  for(auto j=idx;j<ntot;j+=blockDim.x*gridDim.x) {
    auto j = idx; 

    uint32_t pairLayerId=0;
    while(j>=innerLayerCumulativeSize[pairLayerId++]);  --pairLayerId; // move to lower_bound ??

    assert(pairLayerId<nPairs);
    assert(j<innerLayerCumulativeSize[pairLayerId]);
    assert(0==pairLayerId || j>=innerLayerCumulativeSize[pairLayerId-1]);

    uint8_t inner = layerPairs[2*pairLayerId];
    uint8_t outer = layerPairs[2*pairLayerId+1];
    assert(outer>inner);

    auto i = (0==pairLayerId) ? j :  j-innerLayerCumulativeSize[pairLayerId-1];
    i += offsets[inner];

    // printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

    assert(i>=offsets[inner]);
    assert(i<offsets[inner+1]);

    // found hit corresponding to our cuda thread!!!!!
    // do the job

    auto mep = iphi[i];
    auto mez = hh.zg_d[i];
    auto mer = hh.rg_d[i];
    auto cutoff = [&](int j) { return 
        abs(hh.zg_d[j]-mez) > maxz[pairLayerId] ||
      	abs(hh.zg_d[j]-mez) < minz[pairLayerId] ||
        hh.rg_d[j]-mer > maxr[pairLayerId];
    };

    constexpr float z0cut = 12.f;
    auto z0cutoff = [&](int j) {
      auto zo =	hh.zg_d[j];
      auto ro = hh.rg_d[j]; 
      auto dr = ro-mer;
      return dr > maxr[pairLayerId] || 
             dr<0 || std::abs((mez*ro - mer*zo)) > z0cut*dr;
    };

    auto iphicut = phicuts[pairLayerId];

    auto kl = hist[outer].bin(int16_t(mep-iphicut));
    auto kh = hist[outer].bin(int16_t(mep+iphicut));
    auto incr = [](auto & k) { return k = (k+1)%Hist::nbins();};
    int tot  = 0;
    int nmin = 0;
    auto khh = kh;
    incr(khh);
    
    int tooMany=0;
    for (auto kk=kl; kk!=khh; incr(kk)) {
      if (kk!=kl && kk!=kh) nmin+=hist[outer].size(kk);
      for (auto p=hist[outer].begin(kk); p<hist[outer].end(kk); ++p) {
        auto oi=*p;
        assert(oi>=offsets[outer]);
        assert(oi<offsets[outer+1]);

        if (std::min(std::abs(int16_t(iphi[oi]-mep)), std::abs(int16_t(mep-iphi[oi]))) > iphicut)
          continue;
        if (z0cutoff(oi)) continue;
        auto ind = atomicInc(nCells,MaxNumOfDoublets);
        // int layerPairId, int doubletId, int innerHitId,int outerHitId)
        cells[ind].init(hh,pairLayerId,ind,i,oi);
        isOuterHitOfCell[oi].push_back(ind);
        if (isOuterHitOfCell[oi].full()) ++tooMany;
        ++tot;
      }
    }
    if (tooMany>0) printf("OuterHitOfCell full for %d in layer %d/%d, %d:%d   %d,%d\n", i, inner,outer, kl,kh,nmin,tot);

    if (hist[outer].nspills>0)
      printf("spill bin to be checked in %d %d\n",outer,hist[outer].nspills);

    // if (0==hist[outer].nspills) assert(tot>=nmin);
    // look in spill bin as well....


  }  // loop in block...
  }
  __global__
  void getDoubletsFromHisto(GPUCACell * cells, uint32_t * nCells, siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hhp,                
                            GPU::VecArray< unsigned int, 2048> *isOuterHitOfCell) {

    uint8_t const layerPairs[2*13] = {0,1 ,1,2 ,2,3 
                                     // ,0,4 ,1,4 ,2,4 ,4,5 ,5,6  
                                     ,0,7 ,1,7 ,2,7 ,7,8 ,8,9
                                     ,0,4 ,1,4 ,2,4 ,4,5 ,5,6
                                     };

    const int16_t phi0p05 = phi2short(0.05);
    const int16_t phi0p06 = phi2short(0.06);
    const int16_t phi0p07 = phi2short(0.07);

    int16_t const phicuts[13] { phi0p05, phi0p05, phi0p06
                               ,phi0p07, phi0p06, phi0p06, phi0p05, phi0p05
                               ,phi0p07, phi0p06, phi0p06, phi0p05, phi0p05
                              };

    float const minz[13] = { 0., 0., 0.
                            ,0., 0., 0., 0., 0.
      	       	       	    ,0., 0., 0., 0., 0.
                           };

    float const	maxz[13] = { 20.,15.,12.
                            ,30.,20.,20., 50., 50.
       	       	       	    ,30.,20.,20., 50., 50.
                           };

    float const maxr[13] = { 20., 20., 20.
                            ,9., 7., 6., 5., 5.
      	       	       	    ,9., 7., 6., 5., 5.
                           };


    auto const & hh = *hhp;
    doubletsFromHisto(layerPairs, 13, cells, nCells, 
                      hh.iphi_d,hh.hist_d,hh.hitsLayerStart_d,
                      hh, isOuterHitOfCell,
                      phicuts, minz, maxz, maxr);
  }



} // namespace end

#endif // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h
