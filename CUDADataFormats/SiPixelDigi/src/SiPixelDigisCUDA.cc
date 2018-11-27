#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include <cuda_runtime.h>

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t nelements, cuda::stream_t<>& stream) {
  edm::Service<CUDAService> cs;

  xx_d              = cs->make_device_unique<uint16_t[]>(nelements, stream);
  yy_d              = cs->make_device_unique<uint16_t[]>(nelements, stream);
  adc_d             = cs->make_device_unique<uint16_t[]>(nelements, stream);
  moduleInd_d       = cs->make_device_unique<uint16_t[]>(nelements, stream);

  auto view = cs->make_host_unique<DeviceConstView>(stream);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();

  view_d = cs->make_device_unique<DeviceConstView>(stream);
  cudaMemcpyAsync(view_d.get(), view.get(), sizeof(DeviceConstView), cudaMemcpyDefault, stream.id());
}
