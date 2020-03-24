#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "GPUCACell.h"

namespace gpuPixelDoublets {

  //  __device__
  //  __forceinline__
  __global__ void fishbone(GPUCACell::Hits const* __restrict__ hhp,
                           GPUCACell* cells,
                           uint32_t const* __restrict__ nCells,
                           GPUCACell::OuterHitOfCell const* __restrict__ isOuterHitOfCell,
                           uint32_t nHits,
                           bool checkTrack) {
    constexpr auto maxCellsPerHit = GPUCACell::maxCellsPerHit;

    auto const& hh = *hhp;
    // auto layer = [&](uint16_t id) { return hh.cpeParams().layer(id); };

    // x run faster...
    auto firstY = threadIdx.y + blockIdx.y * blockDim.y;
    auto firstX = threadIdx.x;

    float x[maxCellsPerHit], y[maxCellsPerHit], z[maxCellsPerHit], n[maxCellsPerHit];
    uint16_t d[maxCellsPerHit];  // uint8_t l[maxCellsPerHit];
    uint32_t cc[maxCellsPerHit];

    for (int idy = firstY, nt = nHits; idy < nt; idy += gridDim.y * blockDim.y) {
      auto const& vc = isOuterHitOfCell[idy];
      auto s = vc.size();
      if (s < 2)
        continue;
      // if alligned kill one of the two.
      // in principle one could try to relax the cut (only in r-z?) for jumping-doublets
      auto const& c0 = cells[vc[0]];
      auto xo = c0.get_outer_x(hh);
      auto yo = c0.get_outer_y(hh);
      auto zo = c0.get_outer_z(hh);
      auto sg = 0;
      for (int32_t ic = 0; ic < s; ++ic) {
        auto& ci = cells[vc[ic]];
        if (0 == ci.theUsed)
          continue;  // for triplets equivalent to next
        if (checkTrack && ci.tracks().empty())
          continue;
        cc[sg] = vc[ic];
        d[sg] = ci.get_inner_detIndex(hh);
        //      l[sg] = layer(d[sg]);
        x[sg] = ci.get_inner_x(hh) - xo;
        y[sg] = ci.get_inner_y(hh) - yo;
        z[sg] = ci.get_inner_z(hh) - zo;
        n[sg] = x[sg] * x[sg] + y[sg] * y[sg] + z[sg] * z[sg];
        ++sg;
      }
      if (sg < 2)
        continue;
      // here we parallelize
      for (int32_t ic = firstX; ic < sg - 1; ic += blockDim.x) {
        auto& ci = cells[cc[ic]];
        for (auto jc = ic + 1; jc < sg; ++jc) {
          auto& cj = cells[cc[jc]];
          // must be different detectors (in the same layer)
          //        if (d[ic]==d[jc]) continue;
          // || l[ic]!=l[jc]) continue;
          auto cos12 = x[ic] * x[jc] + y[ic] * y[jc] + z[ic] * z[jc];
          if (d[ic] != d[jc] && cos12 * cos12 >= 0.99999f * n[ic] * n[jc]) {
            // alligned:  kill farthest  (prefer consecutive layers)
            if (n[ic] > n[jc]) {
              ci.theDoubletId = -1;
              break;
            } else {
              cj.theDoubletId = -1;
            }
          }
        }  //cj
      }    // ci
    }      // hits
  }
}  // namespace gpuPixelDoublets

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h
