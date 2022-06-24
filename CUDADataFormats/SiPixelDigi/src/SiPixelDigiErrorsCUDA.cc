#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords, SiPixelFormatterErrors errors, cudaStream_t stream)
    : formatterErrors_h(std::move(errors)), nErrorWords_(maxFedWords) {
  assert(maxFedWords != 0);

  memoryPool::Deleter deleter =
      memoryPool::Deleter(std::make_shared<memoryPool::cuda::BundleDelete>(stream, memoryPool::onDevice));
  assert(deleter.pool());

  data_d = memoryPool::cuda::makeBuffer<SiPixelErrorCompact>(maxFedWords, deleter);
  error_d = memoryPool::cuda::makeBuffer<SiPixelErrorCompactVector>(1, deleter);
  error_h = memoryPool::cuda::makeBuffer<SiPixelErrorCompactVector>(1, stream, memoryPool::onHost);

  cudaMemsetAsync(data_d.get(), 0x00, maxFedWords, stream);

  cms::cuda::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
  assert(error_h->empty());
  assert(error_h->capacity() == static_cast<int>(maxFedWords));

  cudaCheck(memoryPool::cuda::copy(error_d, error_h, 1, stream));
}

void SiPixelDigiErrorsCUDA::copyErrorToHostAsync(cudaStream_t stream) {
  cudaCheck(memoryPool::cuda::copy(error_h, error_d, 1, stream));
}

SiPixelDigiErrorsCUDA::HostDataError SiPixelDigiErrorsCUDA::dataErrorToHostAsync(cudaStream_t stream) const {
  // On one hand size() could be sufficient. On the other hand, if
  // someone copies the SimpleVector<>, (s)he might expect the data
  // Buffer to actually have space for capacity() elements.
  auto data = memoryPool::cuda::makeBuffer<SiPixelErrorCompact>(error_h->capacity(), stream, memoryPool::onHost);

  // but transfer only the required amount
  if (not error_h->empty()) {
    cudaCheck(memoryPool::cuda::copy(data, data_d, error_h->size(), stream));
  }
  auto err = *error_h;
  err.set_data(data.get());
  return HostDataError(err, std::move(data));
}
