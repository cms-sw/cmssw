#pragma once

#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"

#include <cstdint>
#include <cstdio>

namespace gpuCalibPixel {

  constexpr uint16_t InvId=9999; // must be > MaxNumModules

  constexpr float VCaltoElectronGain      = 47;   // L2-4: 47 +- 4.7
  constexpr float VCaltoElectronGain_L1   = 50;   // L1:   49.6 +- 2.6
  constexpr float VCaltoElectronOffset    = -60;  // L2-4: -60 +- 130
  constexpr float VCaltoElectronOffset_L1 = -670; // L1:   -670 +- 220


 __global__ void calibADC(uint16_t * id,
			   uint16_t const * x,
			   uint16_t const * y,
			   uint16_t * adc,
			   uint32_t const * moduleStart,
                           SiPixelGainForHLTonGPU const * ped,
                           int numElements
                         )
{


    auto first = moduleStart[1 + blockIdx.x];  
    
    auto me = id[first];
    
    /// depends on "me"

    float conversionFactor = me<96 ? VCaltoElectronGain_L1 : VCaltoElectronGain; 
    float offset =  me<96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset; 
 

#ifdef GPU_DEBUG
    if (me%100==1)
      if (threadIdx.x==0) printf("start pixel calibration for module %d in block %d\n",me,blockIdx.x);
#endif

    first+=threadIdx.x;
 
    // __syncthreads();

    float pedestal=0,gain=0;
    bool isDeadColumn=false, isNoisyColumn=false;
    int oldCol=-1, oldAveragedBlock=-1;

    for (int i=first; i<numElements; i+=blockDim.x) {
       if (id[i]==InvId) continue;  // not valid
       if (id[i]!=me) break;  // end of module
       int row = x[i];
       int col = y[i];
       int averagedBlock = row / ped->numberOfRowsAveragedOver_; // 80....  ( row<80 will be faster...)
       if ( (col!=oldCol) | ( averagedBlock != oldAveragedBlock) ) {
        oldCol=col; oldAveragedBlock= averagedBlock;
        auto ret = ped->getPedAndGain(me,col, row, isDeadColumn, isNoisyColumn);
        pedestal = ret.first; gain = ret.second;
       }
       if ( isDeadColumn | isNoisyColumn ) 
         { id[i]=InvId; adc[i] =0; }
       else {
         float vcal = adc[i] * gain  - pedestal*gain;
         adc[i] = std::max(100, int( vcal * conversionFactor + offset)); 
       }
    } 

 }
}
