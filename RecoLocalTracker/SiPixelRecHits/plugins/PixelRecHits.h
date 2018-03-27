#pragma once

#include<cstdint>
#include<vector>

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

struct context;

struct HitsOnGPU{
   uint32_t * hitsModuleStart_d;
   int32_t  * charge_d;
   float *xg_d, *yg_d, *zg_d;
   float *xerr_d, *yerr_d;
   uint16_t * mr_d;
};

struct HitsOnCPU {
 explicit HitsOnCPU(uint32_t nhits) :
  charge(nhits),xl(nhits),yl(nhits),xe(nhits),ye(nhits), mr(nhits){}
 uint32_t hitsModuleStart[2001];
 std::vector<int32_t> charge;
 std::vector<float> xl, yl;
 std::vector<float> xe, ye;
 std::vector<uint16_t> mr;
};


HitsOnGPU allocHitsOnGPU();

HitsOnCPU pixelRecHits_wrapper(
      context const & c,
      pixelCPEforGPU::ParamsOnGPU const * cpeParams,
      uint32_t ndigis,
      uint32_t nModules, // active modules (with digis)
      HitsOnGPU & hh
);
