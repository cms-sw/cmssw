#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"

#include "GPUCACell.h"
#include "CAConstants.h"


// useful for benchmark
// #define ONLY_PHICUT
// #define USE_ZCUT
// #define NO_CLSCUT

namespace gpuPixelDoublets {

  constexpr uint32_t MaxNumOfDoublets = CAConstants::maxNumberOfDoublets();  // not really relevant

  template<typename Hist>
  __device__
  __forceinline__
  void doubletsFromHisto(uint8_t const * __restrict__ layerPairs,
                         uint32_t nPairs,
                         GPUCACell * cells,
                         uint32_t * nCells,
                         int16_t const * __restrict__ iphi,
                         Hist const & __restrict__ hist,
                         uint32_t const * __restrict__ offsets,
                         siPixelRecHitsHeterogeneousProduct::HitsOnGPU const &  __restrict__ hh,
                         GPUCACell::OuterHitOfCell * isOuterHitOfCell,
                         int16_t const * __restrict__ phicuts,
#ifdef USE_ZCUT
                         float const * __restrict__ minz,
                         float const * __restrict__ maxz,
#endif
                         float const * __restrict__ maxr, bool ideal_cond)
  {

#ifndef NO_CLSCUT 
    // ysize cuts (z in the barrel)  times 8
    constexpr int minYsizeB1=36;
    constexpr int minYsizeB2=28;
    constexpr int maxDYsize12=28;
    constexpr int maxDYsize=20;
#endif

    auto layerSize = [=](uint8_t li) { return offsets[li+1]-offsets[li]; };

    // nPairsMax to be optimized later (originally was 64).
    // If it should be much bigger, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    const int nPairsMax = 16;
    assert(nPairs <= nPairsMax);
    uint32_t innerLayerCumulativeSize[nPairsMax];
    innerLayerCumulativeSize[0] = layerSize(layerPairs[0]);
    for (uint32_t i = 1; i < nPairs; ++i) {
      innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i-1] + layerSize(layerPairs[2*i]);
    }
    auto ntot = innerLayerCumulativeSize[nPairs-1];

    // x runs faster
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    auto first = threadIdx.x;
    auto stride = blockDim.x;
    for (auto j = idy; j < ntot; j += blockDim.y * gridDim.y ) {

      uint32_t pairLayerId=0;
      while (j >= innerLayerCumulativeSize[pairLayerId++]);
      --pairLayerId; // move to lower_bound ??

      assert(pairLayerId < nPairs);
      assert(j < innerLayerCumulativeSize[pairLayerId]);
      assert(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId-1]);

      uint8_t inner = layerPairs[2*pairLayerId];
      uint8_t outer = layerPairs[2*pairLayerId+1];
      assert(outer > inner);

      auto hoff = Hist::histOff(outer);

      auto i = (0 == pairLayerId) ? j : j-innerLayerCumulativeSize[pairLayerId-1];
      i += offsets[inner];

      // printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

      assert(i >= offsets[inner]);
      assert(i < offsets[inner+1]);

      // found hit corresponding to our cuda thread, now do the job
      auto mez = __ldg(hh.zg_d+i);

#ifdef USE_ZCUT
     // this statement is responsible for a 10% slow down of the kernel once all following cuts are optimized...
     if (mez<minz[pairLayerId] || mez>maxz[pairLayerId]) continue;
#endif

#ifndef NO_CLSCUT
      auto mes = __ldg(hh.ysize_d+i);

      // if ideal treat inner ladder as outer
      auto mi = __ldg(hh.detInd_d+i);
      if (inner==0) assert(mi<96);    
      const bool isOuterLadder = ideal_cond ? true : 0 == (mi/8)%2; // only for B1/B2/B3 B4 is opposite, FPIX:noclue...

      // auto mesx = __ldg(hh.xsize_d+i);
      // if (mesx<0) continue; // remove edges in x as overlap will take care

      if (inner==0 && outer>3 && isOuterLadder)  // B1 and F1
         if (mes>0 && mes<minYsizeB1) continue; // only long cluster  (5*8)
      if (inner==1 && outer>3)  // B2 and F1
         if (mes>0 && mes<minYsizeB2) continue;
#endif // NO_CLSCUT

      auto mep = iphi[i];
      auto mer = __ldg(hh.rg_d+i);
 
      constexpr float z0cut = 12.f;                     // cm
      constexpr float hardPtCut = 0.5f;                 // GeV
      constexpr float minRadius = hardPtCut * 87.78f;   // cm (1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field)
      constexpr float minRadius2T4 = 4.f*minRadius*minRadius;
      auto ptcut = [&](int j) {
        auto r2t4 = minRadius2T4;
        auto ri = mer;
        auto ro = __ldg(hh.rg_d+j);
        auto dphi = short2phi( min( abs(int16_t(mep-iphi[j])), abs(int16_t(iphi[j]-mep)) ) );
        return dphi*dphi * (r2t4 - ri*ro) > (ro-ri)*(ro-ri);
      };
      auto z0cutoff = [&](int j) {
        auto zo = __ldg(hh.zg_d+j);
        auto ro = __ldg(hh.rg_d+j);
        auto dr = ro-mer;
        return dr > maxr[pairLayerId] ||
          dr<0 || std::abs((mez*ro - mer*zo)) > z0cut*dr;
      };

#ifndef NO_CLSCUT
      auto zsizeCut = [&](int j) {
        auto onlyBarrel = outer<4;
        auto so = __ldg(hh.ysize_d+j);
        //auto sox = __ldg(hh.xsize_d+j);
        auto dy = inner==0 ? ( isOuterLadder ? maxDYsize12: 100 ) : maxDYsize;
        return onlyBarrel && mes>0 && so>0 && std::abs(so-mes)>dy;
      };
#endif

      auto iphicut = phicuts[pairLayerId];

      auto kl = Hist::bin(int16_t(mep-iphicut));
      auto kh = Hist::bin(int16_t(mep+iphicut));
      auto incr = [](auto & k) { return k = (k+1) % Hist::nbins();};

#ifdef GPU_DEBUG
      int  tot  = 0;
      int  nmin = 0;
      int tooMany=0;
#endif

      auto khh = kh;
      incr(khh);
      for (auto kk = kl; kk != khh; incr(kk)) {
#ifdef GPU_DEBUG
        if (kk != kl && kk != kh)
          nmin += hist.size(kk+hoff);
#endif
        auto const * __restrict__ p = hist.begin(kk+hoff);
        auto const * __restrict__ e = hist.end(kk+hoff);
        p+=first;
        for (;p < e; p+=stride) {
          auto oi=__ldg(p);
          assert(oi>=offsets[outer]);
          assert(oi<offsets[outer+1]);

          if (std::min(std::abs(int16_t(iphi[oi]-mep)), std::abs(int16_t(mep-iphi[oi]))) > iphicut)
            continue;
#ifndef ONLY_PHICUT
#ifndef NO_CLSCUT
          if (zsizeCut(oi)) continue;
#endif
          if (z0cutoff(oi) || ptcut(oi)) continue;
#endif
          auto ind = atomicAdd(nCells, 1); 
          if (ind>=MaxNumOfDoublets) {atomicSub(nCells, 1); break; } // move to SimpleVector??
          // int layerPairId, int doubletId, int innerHitId, int outerHitId)
          cells[ind].init(hh, pairLayerId, ind, i, oi);
          isOuterHitOfCell[oi].push_back(ind);
#ifdef GPU_DEBUG
          if (isOuterHitOfCell[oi].full()) ++tooMany;
          ++tot;
#endif
        }
      }
#ifdef GPU_DEBUG
      if (tooMany > 0)
        printf("OuterHitOfCell full for %d in layer %d/%d, %d,%d %d\n", i, inner, outer, nmin, tot, tooMany);
#endif
    }  // loop in block...
  }

  constexpr auto getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
  constexpr auto getDoubletsFromHistoMinBlocksPerMP = 16;

  __global__
  __launch_bounds__(getDoubletsFromHistoMaxBlockSize,getDoubletsFromHistoMinBlocksPerMP)
  void getDoubletsFromHisto(GPUCACell * cells,
                            uint32_t * nCells,
                            siPixelRecHitsHeterogeneousProduct::HitsOnGPU const *  __restrict__ hhp,
                            GPUCACell::OuterHitOfCell * isOuterHitOfCell,
                            bool ideal_cond)
  {
    constexpr int nPairs = 13;
    constexpr const uint8_t layerPairs[2*nPairs] = {
      0, 1,  1, 2,  2, 3,
      // 0, 4,  1, 4,  2, 4,  4, 5,  5, 6,
      0, 7,  1, 7,  2, 7,  7, 8,  8, 9, // neg
      0, 4,  1, 4,  2, 4,  4, 5,  5, 6,  // pos
    };

    constexpr int16_t phi0p05 = 522;    // round(521.52189...) = phi2short(0.05);
    constexpr int16_t phi0p06 = 626;    // round(625.82270...) = phi2short(0.06);
    constexpr int16_t phi0p07 = 730;    // round(730.12648...) = phi2short(0.07);

    constexpr const int16_t phicuts[nPairs] {
      phi0p05, phi0p05, phi0p06,
      phi0p07, phi0p06, phi0p06, phi0p05, phi0p05,
      phi0p07, phi0p06, phi0p06, phi0p05, phi0p05
    };

#ifdef USE_ZCUT
    float const minz[nPairs] = {
      -20., -22., -22.,
      -30., -30.,-30., -70., -70.,
        0., 10., 15., -70., -70.
    };

    float const maxz[nPairs] = {
      20., 22., 22.,
       0., -10., -15., 70., 70.,
      30., 30., 30., 70., 70.
    };
#endif

    float const maxr[nPairs] = {
      20., 20., 20.,
       9.,  7.,  6.,  5.,  5.,
       9.,  7.,  6.,  5.,  5.
    };

    auto const &  __restrict__ hh = *hhp;
    doubletsFromHisto(layerPairs, nPairs, cells, nCells,
                      hh.iphi_d, *hh.hist_d, hh.hitsLayerStart_d,
                      hh, isOuterHitOfCell,
                      phicuts, 
#ifdef USE_ZCUT
                      minz, maxz, 
#endif
                      maxr , ideal_cond);
  }



} // namespace end

#endif // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h
