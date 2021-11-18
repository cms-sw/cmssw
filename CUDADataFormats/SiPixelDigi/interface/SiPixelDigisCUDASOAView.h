#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDASOAView_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDASOAView_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

class SiPixelDigisCUDASOAView {
public:
  enum class StorageLocation { XX = 0, YY = 1, MODULEIND = 2, ADC = 3, CLUS = 4, PDIGI = 6, RAWIDARR = 8, MAX = 10 };
  /*
  ============================================================================================================================
  |    XX     |     YY    | MODULEIND |    ADC    |          CLUS         |          PDIGI         |         RAWIDARR        |
  ============================================================================================================================
  |  0: N*16  |  1: N*16  |  2: N*16  |  3: N*16  |        4: N*32        |         6: N*32        |         8: N*32         |
  ============================================================================================================================
  */
  // These are for CPU output
  // we don't copy local x and y coordinates and module index
  enum class StorageLocationHost { ADC = 0, CLUS = 1, PDIGI = 3, RAWIDARR = 5, MAX = 7 };
  /*
  ========================================================================================
  |    ADC    |          CLUS         |          PDIGI         |         RAWIDARR        |
  ========================================================================================
  |  0: N*16  |        1: N*32        |         3: N*32        |         5: N*32         |
  ========================================================================================
  */

  __device__ __forceinline__ uint16_t xx(int i) const { return __ldg(xx_ + i); }
  __device__ __forceinline__ uint16_t yy(int i) const { return __ldg(yy_ + i); }
  __device__ __forceinline__ uint16_t adc(int i) const { return __ldg(adc_ + i); }
  __device__ __forceinline__ uint16_t moduleInd(int i) const { return __ldg(moduleInd_ + i); }
  __device__ __forceinline__ int32_t clus(int i) const { return __ldg(clus_ + i); }
  __device__ __forceinline__ uint32_t pdigi(int i) const { return __ldg(pdigi_ + i); }
  __device__ __forceinline__ uint32_t rawIdArr(int i) const { return __ldg(rawIdArr_ + i); }

  uint16_t *xx_;  // local coordinates of each pixel
  uint16_t *yy_;
  uint16_t *adc_;        // ADC of each pixel
  uint16_t *moduleInd_;  // module id of each pixel
  int32_t *clus_;        // cluster id of each pixel
  uint32_t *pdigi_;
  uint32_t *rawIdArr_;
};

#endif