#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <cstdint>
#include <cstdio>

#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

namespace gpuCalibPixel {

  constexpr uint16_t InvId=9999; // must be > MaxNumModules

  constexpr float VCaltoElectronGain      = 47;   // L2-4: 47 +- 4.7
  constexpr float VCaltoElectronGain_L1   = 50;   // L1:   49.6 +- 2.6
  constexpr float VCaltoElectronOffset    = -60;  // L2-4: -60 +- 130
  constexpr float VCaltoElectronOffset_L1 = -670; // L1:   -670 +- 220


 __global__ void calibDigis(uint16_t * id,
                           uint16_t const * __restrict__ x,
                           uint16_t const * __restrict__ y,
                           uint16_t * adc,
                           SiPixelGainForHLTonGPU const * __restrict__ ped,
                           int numElements
                         )
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numElements) return;
    if (InvId==id[i]) return;

    float conversionFactor = id[i]<96 ? VCaltoElectronGain_L1 : VCaltoElectronGain;
    float offset =  id[i]<96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset;

    bool isDeadColumn=false, isNoisyColumn=false;
 
    int row = x[i];
    int col = y[i];
    auto ret = ped->getPedAndGain(id[i], col, row, isDeadColumn, isNoisyColumn);
    float pedestal = ret.first; float gain = ret.second;
    // float pedestal = 0; float gain = 1.;
    if ( isDeadColumn | isNoisyColumn )
      { 
        id[i]=InvId; adc[i] =0; 
        printf("bad pixel at %d in %d\n",i,id[i]);
    }
    else {
      float vcal = adc[i] * gain  - pedestal*gain;
      adc[i] = std::max(100, int( vcal * conversionFactor + offset));
    }

    // if (threadIdx.x==0)
    //  printf ("calibrated %d\n",id[i]);
}

 __global__ void calibADCByModule(uint16_t * id,
			   uint16_t const * __restrict__ x,
			   uint16_t const * __restrict__ y,
			   uint16_t * adc,
			   uint32_t * moduleStart,
                           SiPixelGainForHLTonGPU const * __restrict__ ped,
                           int numElements
                         )
{


    auto first = moduleStart[1 + blockIdx.x];  
    
    auto me = id[first];
    
    assert(me<2000);

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

    __syncthreads(); 
    //reset start
    if(0==threadIdx.x) {
     auto & k = moduleStart[1 + blockIdx.x];
     while (id[k]==InvId) ++k;
    }
     

 }


}

#endif // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
