#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include "CUDADataFormats/Common/interface/device_unique_ptr.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <cuda/api_wrappers.h>

class SiPixelDigisCUDA {
public:
  SiPixelDigisCUDA() = default;
  explicit SiPixelDigisCUDA(size_t nelements, cuda::stream_t<>& stream);
  ~SiPixelDigisCUDA() = default;

  SiPixelDigisCUDA(const SiPixelDigisCUDA&) = delete;
  SiPixelDigisCUDA& operator=(const SiPixelDigisCUDA&) = delete;
  SiPixelDigisCUDA(SiPixelDigisCUDA&&) = default;
  SiPixelDigisCUDA& operator=(SiPixelDigisCUDA&&) = default;

  uint16_t * xx() { return xx_d.get(); }
  uint16_t * yy() { return yy_d.get(); }
  uint16_t * adc() { return adc_d.get(); }
  uint16_t * moduleInd() { return moduleInd_d.get(); }

  uint16_t const *xx() const { return xx_d.get(); }
  uint16_t const *yy() const { return yy_d.get(); }
  uint16_t const *adc() const { return adc_d.get(); }
  uint16_t const *moduleInd() const { return moduleInd_d.get(); }

  uint16_t const *c_xx() const { return xx_d.get(); }
  uint16_t const *c_yy() const { return yy_d.get(); }
  uint16_t const *c_adc() const { return adc_d.get(); }
  uint16_t const *c_moduleInd() const { return moduleInd_d.get(); }

  class DeviceConstView {
  public:
    DeviceConstView() = default;

#ifdef __CUDACC__
    __device__ __forceinline__ uint16_t xx(int i) const { return __ldg(xx_+i); }
    __device__ __forceinline__ uint16_t yy(int i) const { return __ldg(yy_+i); }
    __device__ __forceinline__ uint16_t adc(int i) const { return __ldg(adc_+i); }
    __device__ __forceinline__ uint16_t moduleInd(int i) const { return __ldg(moduleInd_+i); }
#endif

    friend class SiPixelDigisCUDA;

  private:
    uint16_t const *xx_ = nullptr;
    uint16_t const *yy_ = nullptr;
    uint16_t const *adc_ = nullptr;
    uint16_t const *moduleInd_ = nullptr;
  };

  const DeviceConstView *view() const { return view_d.get(); }

private:
  edm::cuda::device::unique_ptr<uint16_t[]> xx_d;        // local coordinates of each pixel
  edm::cuda::device::unique_ptr<uint16_t[]> yy_d;        //
  edm::cuda::device::unique_ptr<uint16_t[]> adc_d;       // ADC of each pixel
  edm::cuda::device::unique_ptr<uint16_t[]> moduleInd_d; // module id of each pixel
  edm::cuda::device::unique_ptr<DeviceConstView> view_d; // "me" pointer
};

#endif
