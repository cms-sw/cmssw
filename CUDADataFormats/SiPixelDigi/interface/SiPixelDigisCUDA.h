#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include <cuda/api_wrappers.h>

class SiPixelDigisCUDA {
public:
  SiPixelDigisCUDA() = default;
  explicit SiPixelDigisCUDA(size_t maxFedWords, cuda::stream_t<>& stream);
  ~SiPixelDigisCUDA() = default;

  SiPixelDigisCUDA(const SiPixelDigisCUDA&) = delete;
  SiPixelDigisCUDA& operator=(const SiPixelDigisCUDA&) = delete;
  SiPixelDigisCUDA(SiPixelDigisCUDA&&) = default;
  SiPixelDigisCUDA& operator=(SiPixelDigisCUDA&&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

  uint16_t * xx() { return xx_d.get(); }
  uint16_t * yy() { return yy_d.get(); }
  uint16_t * adc() { return adc_d.get(); }
  uint16_t * moduleInd() { return moduleInd_d.get(); }
  int32_t  * clus() { return clus_d.get(); }
  uint32_t * pdigi() { return pdigi_d.get(); }
  uint32_t * rawIdArr() { return rawIdArr_d.get(); }

  uint16_t const *xx() const { return xx_d.get(); }
  uint16_t const *yy() const { return yy_d.get(); }
  uint16_t const *adc() const { return adc_d.get(); }
  uint16_t const *moduleInd() const { return moduleInd_d.get(); }
  int32_t  const *clus() const { return clus_d.get(); } 
  uint32_t const *pdigi() const { return pdigi_d.get(); }
  uint32_t const *rawIdArr() const { return rawIdArr_d.get(); }

  uint16_t const *c_xx() const { return xx_d.get(); }
  uint16_t const *c_yy() const { return yy_d.get(); }
  uint16_t const *c_adc() const { return adc_d.get(); }
  uint16_t const *c_moduleInd() const { return moduleInd_d.get(); }
  int32_t  const *c_clus() const { return clus_d.get(); }
  uint32_t const *c_pdigi() const { return pdigi_d.get(); }
  uint32_t const *c_rawIdArr() const { return rawIdArr_d.get(); }
  
  cudautils::host::unique_ptr<uint16_t[]> adcToHostAsync(cuda::stream_t<>& stream) const;
  cudautils::host::unique_ptr< int32_t[]> clusToHostAsync(cuda::stream_t<>& stream) const;
  cudautils::host::unique_ptr<uint32_t[]> pdigiToHostAsync(cuda::stream_t<>& stream) const;
  cudautils::host::unique_ptr<uint32_t[]> rawIdArrToHostAsync(cuda::stream_t<>& stream) const;

  class DeviceConstView {
  public:
    DeviceConstView() = default;

#ifdef __CUDACC__
    __device__ __forceinline__ uint16_t xx(int i) const { return __ldg(xx_+i); }
    __device__ __forceinline__ uint16_t yy(int i) const { return __ldg(yy_+i); }
    __device__ __forceinline__ uint16_t adc(int i) const { return __ldg(adc_+i); }
    __device__ __forceinline__ uint16_t moduleInd(int i) const { return __ldg(moduleInd_+i); }
    __device__ __forceinline__ int32_t  clus(int i) const { return __ldg(clus_+i); }
#endif

    friend class SiPixelDigisCUDA;

  private:
    uint16_t const *xx_;
    uint16_t const *yy_;
    uint16_t const *adc_;
    uint16_t const *moduleInd_;
    int32_t  const *clus_;
  };

  const DeviceConstView *view() const { return view_d.get(); }

private:
  // These are consumed by downstream device code
  cudautils::device::unique_ptr<uint16_t[]> xx_d;        // local coordinates of each pixel
  cudautils::device::unique_ptr<uint16_t[]> yy_d;        //
  cudautils::device::unique_ptr<uint16_t[]> adc_d;       // ADC of each pixel
  cudautils::device::unique_ptr<uint16_t[]> moduleInd_d; // module id of each pixel
  cudautils::device::unique_ptr<int32_t[]>  clus_d;      // cluster id of each pixel
  cudautils::device::unique_ptr<DeviceConstView> view_d; // "me" pointer

  // These are for CPU output; should we (eventually) place them to a
  // separate product?
  cudautils::device::unique_ptr<uint32_t[]> pdigi_d;
  cudautils::device::unique_ptr<uint32_t[]> rawIdArr_d;

  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif
