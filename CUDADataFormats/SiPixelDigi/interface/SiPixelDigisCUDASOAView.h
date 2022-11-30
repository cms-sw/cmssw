#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDASOAView_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDASOAView_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include <cstdint>

class SiPixelDigisCUDASOAView {
public:
  friend class SiPixelDigisCUDA;

  template <typename TrackerTraits>
  friend class SiPixelRecHitSoAFromLegacyT;

  enum class StorageLocation {
    kCLUS = 0,
    kPDIGI = 2,
    kRAWIDARR = 4,
    kADC = 6,
    kXX = 7,
    kYY = 8,
    kMODULEIND = 9,
    kMAX = 10
  };
  /*
  ============================================================================================================================
  |          CLUS         |          PDIGI         |         RAWIDARR        |    ADC    |    XX     |     YY    | MODULEIND |
  ============================================================================================================================
  |        0: N*32        |         2: N*32        |         4: N*32         |  6: N*16  |  7: N*16  |  8: N*16  |  9: N*16  |
  ============================================================================================================================
  */
  // These are for CPU output
  // we don't copy local x and y coordinates and module index
  enum class StorageLocationHost { kCLUS = 0, kPDIGI = 2, kRAWIDARR = 4, kADC = 6, kMAX = 7 };
  /*
  ========================================================================================
  |          CLUS         |          PDIGI         |         RAWIDARR        |    ADC    |
  ========================================================================================
  |        0: N*32        |         2: N*32        |         4: N*32         |  6: N*16  |
  ========================================================================================
  */

  SiPixelDigisCUDASOAView() = default;

  template <typename StoreType>
  SiPixelDigisCUDASOAView(StoreType& store, int maxFedWords, StorageLocation s) {
    xx_ = getColumnAddress<uint16_t>(StorageLocation::kXX, store, maxFedWords);
    yy_ = getColumnAddress<uint16_t>(StorageLocation::kYY, store, maxFedWords);
    adc_ = getColumnAddress<uint16_t>(StorageLocation::kADC, store, maxFedWords);
    moduleInd_ = getColumnAddress<uint16_t>(StorageLocation::kMODULEIND, store, maxFedWords);
    clus_ = getColumnAddress<int32_t>(StorageLocation::kCLUS, store, maxFedWords);
    pdigi_ = getColumnAddress<uint32_t>(StorageLocation::kPDIGI, store, maxFedWords);
    rawIdArr_ = getColumnAddress<uint32_t>(StorageLocation::kRAWIDARR, store, maxFedWords);
  }

  template <typename StoreType>
  SiPixelDigisCUDASOAView(StoreType& store, int maxFedWords, StorageLocationHost s) {
    adc_ = getColumnAddress<uint16_t>(StorageLocationHost::kADC, store, maxFedWords);
    clus_ = getColumnAddress<int32_t>(StorageLocationHost::kCLUS, store, maxFedWords);
    pdigi_ = getColumnAddress<uint32_t>(StorageLocationHost::kPDIGI, store, maxFedWords);
    rawIdArr_ = getColumnAddress<uint32_t>(StorageLocationHost::kRAWIDARR, store, maxFedWords);
  }

  __device__ __forceinline__ uint16_t xx(int i) const { return __ldg(xx_ + i); }
  __device__ __forceinline__ uint16_t yy(int i) const { return __ldg(yy_ + i); }
  __device__ __forceinline__ uint16_t adc(int i) const { return __ldg(adc_ + i); }
  __device__ __forceinline__ uint16_t moduleInd(int i) const { return __ldg(moduleInd_ + i); }
  __device__ __forceinline__ int32_t clus(int i) const { return __ldg(clus_ + i); }
  __device__ __forceinline__ uint32_t pdigi(int i) const { return __ldg(pdigi_ + i); }
  __device__ __forceinline__ uint32_t rawIdArr(int i) const { return __ldg(rawIdArr_ + i); }

  const uint16_t* xx() const { return xx_; }
  const uint16_t* yy() const { return yy_; }
  const uint16_t* adc() const { return adc_; }
  const uint16_t* moduleInd() const { return moduleInd_; }
  const int32_t* clus() const { return clus_; }
  const uint32_t* pdigi() const { return pdigi_; }
  const uint32_t* rawIdArr() const { return rawIdArr_; }

  uint16_t* xx() { return xx_; }
  uint16_t* yy() { return yy_; }
  uint16_t* adc() { return adc_; }
  uint16_t* moduleInd() { return moduleInd_; }
  int32_t* clus() { return clus_; }
  uint32_t* pdigi() { return pdigi_; }
  uint32_t* rawIdArr() { return rawIdArr_; }

private:
  uint16_t* xx_;  // local coordinates of each pixel
  uint16_t* yy_;
  uint16_t* adc_;        // ADC of each pixel
  uint16_t* moduleInd_;  // module id of each pixel
  int32_t* clus_;        // cluster id of each pixel
  uint32_t* pdigi_;
  uint32_t* rawIdArr_;

  template <typename ReturnType, typename StoreType, typename LocationType>
  ReturnType* getColumnAddress(LocationType column, StoreType& store, int size) {
    return reinterpret_cast<ReturnType*>(store.get() + static_cast<int>(column) * roundFor128ByteAlignment(size));
  }

  static int roundFor128ByteAlignment(int size) {
    constexpr int mul = 128 / sizeof(uint16_t);
    return ((size + mul - 1) / mul) * mul;
  };
};

#endif
