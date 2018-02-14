#pragma once

#include<cstdint>

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

struct context;

struct HitsOnGPU{

   uint32_t * hitsModuleStart_d;
   int32_t  * charge_d;
   float *xg_d, *yg_d, *zg_d;
};

HitsOnGPU allocHitsOnGPU();

void pixelRecHits_wrapper(
      context const & c,
      pixelCPEforGPU::ParamsOnGPU const * cpeParams,
      uint32_t ndigis,
      uint32_t nModules, // active modules (with digis)
      HitsOnGPU & hh
);
