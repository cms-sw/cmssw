#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"

SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cuda::stream_t<>& stream):
  formatterErrors_h(std::move(errors))
{
  edm::Service<CUDAService> cs;

  error_d = cs->make_device_unique<GPU::SimpleVector<PixelErrorCompact>>(stream);
  data_d = cs->make_device_unique<PixelErrorCompact[]>(maxFedWords, stream);

  cudautils::memsetAsync(data_d, 0x00, maxFedWords, stream);

  error_h = cs->make_host_unique<GPU::SimpleVector<PixelErrorCompact>>(stream);
  GPU::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
  assert(error_h->size() == 0);
  assert(error_h->capacity() == static_cast<int>(maxFedWords));

  cudautils::copyAsync(error_d, error_h, stream);
}

void SiPixelDigiErrorsCUDA::copyErrorToHostAsync(cuda::stream_t<>& stream) {
  cudautils::copyAsync(error_h, error_d, stream);
}

SiPixelDigiErrorsCUDA::HostDataError SiPixelDigiErrorsCUDA::dataErrorToHostAsync(cuda::stream_t<>& stream) const {
  edm::Service<CUDAService> cs;
  // On one hand size() could be sufficient. On the other hand, if
  // someone copies the SimpleVector<>, (s)he might expect the data
  // buffer to actually have space for capacity() elements.
  auto data = cs->make_host_unique<PixelErrorCompact[]>(error_h->capacity(), stream);

  // but transfer only the required amount
  if(error_h->size() > 0) {
    cudautils::copyAsync(data, data_d, error_h->size(), stream);
  }
  auto err = *error_h;
  err.set_data(data.get());
  return HostDataError(std::move(err), std::move(data));
}
