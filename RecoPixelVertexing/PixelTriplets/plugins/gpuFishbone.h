#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "DataFormats/Math/interface/approx_atan2.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"
#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"

#include "GPUCACell.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"

namespace gpuPixelDoublets {

//  __device__
//  __forceinline__
  __global__
  void fishbone(
               GPUCACell::Hits const *  __restrict__ hhp,
               GPUCACell * cells, uint32_t const * __restrict__ nCells,
               GPUCACell::OuterHitOfCell const * __restrict__ isOuterHitOfCell,
               uint32_t nHits,
               uint32_t stride, bool checkTrack) {

    constexpr auto maxCellsPerHit = GPUCACell::maxCellsPerHit;


    auto const & hh = *hhp;
    uint8_t const * __restrict__ layerp =  hh.phase1TopologyLayer_d;
    auto layer = [&](uint16_t id) { return __ldg(layerp+id/phase1PixelTopology::maxModuleStride);};

    auto ldx = threadIdx.x + blockIdx.x * blockDim.x;
    auto idx = ldx/stride;
    auto first = ldx - idx*stride;
    assert(first<stride);

    if (idx>=nHits) return;
    auto const & vc = isOuterHitOfCell[idx];
    auto s = vc.size();
    if (s<2) return;
    // if alligned kill one of the two.
    auto const & c0 = cells[vc[0]];
    auto xo = c0.get_outer_x(hh);
    auto yo = c0.get_outer_y(hh);
    auto zo = c0.get_outer_z(hh);
    float x[maxCellsPerHit], y[maxCellsPerHit],z[maxCellsPerHit], n[maxCellsPerHit];
    uint16_t d[maxCellsPerHit]; // uint8_t l[maxCellsPerHit];
    uint32_t cc[maxCellsPerHit];
    auto sg=0;
    for (uint32_t ic=0; ic<s; ++ic) {
      auto & ci = cells[vc[ic]];
      if (checkTrack && 0==ci.theTracks.size()) continue;
      cc[sg] = vc[ic];
      d[sg] = ci.get_inner_detId(hh);
//      l[sg] = layer(d[sg]);
      x[sg] = ci.get_inner_x(hh) -xo;
      y[sg] = ci.get_inner_y(hh) -yo;
      z[sg] = ci.get_inner_z(hh) -zo;
      n[sg] = x[sg]*x[sg]+y[sg]*y[sg]+z[sg]*z[sg];
      ++sg;
    }
    if (sg<2) return;   
    // here we parallelize
    for (uint32_t ic=first; ic<sg-1;  ic+=stride) {
      auto & ci = cells[cc[ic]];
      for    (auto jc=ic+1; jc<sg; ++jc) {
        auto & cj = cells[cc[jc]];
        // must be different detectors (in the same layer)
//        if (d[ic]==d[jc]) continue;
        // || l[ic]!=l[jc]) continue;
        auto cos12 = x[ic]*x[jc]+y[ic]*y[jc]+z[ic]*z[jc];
        if (d[ic]!=d[jc] && cos12*cos12 >= 0.99999f*n[ic]*n[jc]) {
         // alligned:  kill farthest  (prefer consecutive layers)
         if (n[ic]>n[jc]) {
           ci.theDoubletId=-1; 
           break;
         } else {
           cj.theDoubletId=-1;
         }
        }
      } //cj   
    } // ci
  }

}

#endif
