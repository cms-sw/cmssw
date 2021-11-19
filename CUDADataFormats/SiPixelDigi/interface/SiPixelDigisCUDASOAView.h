#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDASOAView_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDASOAView_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

class SiPixelDigisCUDASOAView {
public:
  enum class StorageLocation { CLUS = 0, PDIGI = 2, RAWIDARR = 4, ADC = 6, XX = 7, YY = 8, MODULEIND = 9, MAX = 10 };
  /*
  ============================================================================================================================
  |          CLUS         |          PDIGI         |         RAWIDARR        |    ADC    |    XX     |     YY    | MODULEIND |
  ============================================================================================================================
  |        0: N*32        |         2: N*32        |         4: N*32         |  6: N*16  |  7: N*16  |  8: N*16  |  9: N*16  |
  ============================================================================================================================
  */
  // These are for CPU output
  // we don't copy local x and y coordinates and module index
  enum class StorageLocationHost { CLUS = 0, PDIGI = 2, RAWIDARR = 4, ADC = 6, MAX = 7 };
  /*
  ========================================================================================
  |          CLUS         |          PDIGI         |         RAWIDARR        |    ADC    |
  ========================================================================================
  |        0: N*32        |         2: N*32        |         4: N*32         |  6: N*16  |
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