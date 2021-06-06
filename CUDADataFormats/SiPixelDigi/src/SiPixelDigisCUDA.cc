#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
    : xx_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream)),
      yy_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream)),
      adc_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream)),
      moduleInd_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream)),
      clus_d(cms::cuda::make_device_unique<int32_t[]>(maxFedWords, stream)),
      view_d(cms::cuda::make_device_unique<DeviceConstView>(stream)),
      pdigi_d(cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream)),
      rawIdArr_d(cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream)) {
  auto view = cms::cuda::make_host_unique<DeviceConstView>(stream);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();
  cms::cuda::copyAsync(view_d, view, stream);
}

cms::cuda::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<int32_t[]> SiPixelDigisCUDA::clusToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, clus_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::pdigiToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, pdigi_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::rawIdArrToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, rawIdArr_d, nDigis(), stream);
  return ret;
}
