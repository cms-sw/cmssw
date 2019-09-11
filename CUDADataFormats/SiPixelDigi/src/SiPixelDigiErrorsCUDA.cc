#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"

#include <cassert>

SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cuda::stream_t<>& stream)
    : formatterErrors_h(std::move(errors)) {
  error_d = cudautils::make_device_unique<GPU::SimpleVector<PixelErrorCompact>>(stream);
  data_d = cudautils::make_device_unique<PixelErrorCompact[]>(maxFedWords, stream);

  cudautils::memsetAsync(data_d, 0x00, maxFedWords, stream);

  error_h = cudautils::make_host_unique<GPU::SimpleVector<PixelErrorCompact>>(stream);
  GPU::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
  assert(error_h->empty());
  assert(error_h->capacity() == static_cast<int>(maxFedWords));

  cudautils::copyAsync(error_d, error_h, stream);
}

void SiPixelDigiErrorsCUDA::copyErrorToHostAsync(cuda::stream_t<>& stream) {
  cudautils::copyAsync(error_h, error_d, stream);
}

SiPixelDigiErrorsCUDA::HostDataError SiPixelDigiErrorsCUDA::dataErrorToHostAsync(cuda::stream_t<>& stream) const {
  // On one hand size() could be sufficient. On the other hand, if
  // someone copies the SimpleVector<>, (s)he might expect the data
  // buffer to actually have space for capacity() elements.
  auto data = cudautils::make_host_unique<PixelErrorCompact[]>(error_h->capacity(), stream);

  // but transfer only the required amount
  if (not error_h->empty()) {
    cudautils::copyAsync(data, data_d, error_h->size(), stream);
  }
  auto err = *error_h;
  err.set_data(data.get());
  return HostDataError(std::move(err), std::move(data));
}
