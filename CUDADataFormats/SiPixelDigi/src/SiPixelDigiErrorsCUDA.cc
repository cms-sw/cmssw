#include <cassert>

#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"

SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords, SiPixelFormatterErrors errors, cudaStream_t stream)
    : data_d(cms::cuda::make_device_unique<SiPixelErrorCompact[]>(maxFedWords, stream)),
      error_d(cms::cuda::make_device_unique<SiPixelErrorCompactVector>(stream)),
      error_h(cms::cuda::make_host_unique<SiPixelErrorCompactVector>(stream)),
      formatterErrors_h(std::move(errors)) {
  cms::cuda::memsetAsync(data_d, 0x00, maxFedWords, stream);

  cms::cuda::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
  assert(error_h->empty());
  assert(error_h->capacity() == static_cast<int>(maxFedWords));

  cms::cuda::copyAsync(error_d, error_h, stream);
}

void SiPixelDigiErrorsCUDA::copyErrorToHostAsync(cudaStream_t stream) {
  cms::cuda::copyAsync(error_h, error_d, stream);
}

SiPixelDigiErrorsCUDA::HostDataError SiPixelDigiErrorsCUDA::dataErrorToHostAsync(cudaStream_t stream) const {
  // On one hand size() could be sufficient. On the other hand, if
  // someone copies the SimpleVector<>, (s)he might expect the data
  // buffer to actually have space for capacity() elements.
  auto data = cms::cuda::make_host_unique<SiPixelErrorCompact[]>(error_h->capacity(), stream);

  // but transfer only the required amount
  if (not error_h->empty()) {
    cms::cuda::copyAsync(data, data_d, error_h->size(), stream);
  }
  auto err = *error_h;
  err.set_data(data.get());
  return HostDataError(err, std::move(data));
}
