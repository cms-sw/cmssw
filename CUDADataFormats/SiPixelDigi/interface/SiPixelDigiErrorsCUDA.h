#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include <cuda_runtime.h>

#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memoryPool.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"


class SiPixelDigiErrorsCUDA {
public:
  using SiPixelErrorCompactVector = cms::cuda::SimpleVector<SiPixelErrorCompact>;

  SiPixelDigiErrorsCUDA() = default;
  inline SiPixelDigiErrorsCUDA(size_t maxFedWords, SiPixelFormatterErrors errors, cudaStream_t stream);
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const SiPixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  SiPixelErrorCompactVector* error() { return error_d.get(); }
  SiPixelErrorCompactVector const* error() const { return error_d.get(); }

  using HostDataError = std::pair<SiPixelErrorCompactVector, memoryPool::buffer<SiPixelErrorCompact>>;
  inline HostDataError dataErrorToHostAsync(cudaStream_t stream) const;

  inline void copyErrorToHostAsync(cudaStream_t stream);
  int nErrorWords() const { return nErrorWords_; }

private:
  memoryPool::buffer<SiPixelErrorCompact> data_d;
  memoryPool::buffer<SiPixelErrorCompactVector> error_d;
  memoryPool::buffer<SiPixelErrorCompactVector> error_h;
  SiPixelFormatterErrors formatterErrors_h;
  int nErrorWords_ = 0;
};


#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords, SiPixelFormatterErrors errors, cudaStream_t stream) :
      formatterErrors_h(std::move(errors)),
      nErrorWords_(maxFedWords) {
  assert(maxFedWords != 0);

  memoryPool::Deleter deleter = memoryPool::Deleter(std::make_shared<memoryPool::cuda::BundleDelete>(stream, memoryPool::onDevice));
  assert(deleter.pool());

  data_d =  memoryPool::cuda::make_buffer<SiPixelErrorCompact>(maxFedWords, deleter);
  error_d = memoryPool::cuda::make_buffer<SiPixelErrorCompactVector>(1,deleter);
  error_h = memoryPool::cuda::make_buffer<SiPixelErrorCompactVector>(1,stream,memoryPool::onHost);


  cudaMemsetAsync(data_d.get(), 0x00, maxFedWords, stream);

  cms::cuda::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
  assert(error_h->empty());
  assert(error_h->capacity() == static_cast<int>(maxFedWords));

  cudaCheck(memoryPool::cuda::copy(error_d, error_h, 1,stream));
}

void SiPixelDigiErrorsCUDA::copyErrorToHostAsync(cudaStream_t stream) {
  cudaCheck(memoryPool::cuda::copy(error_h, error_d, 1,stream));
}

SiPixelDigiErrorsCUDA::HostDataError SiPixelDigiErrorsCUDA::dataErrorToHostAsync(cudaStream_t stream) const {
  // On one hand size() could be sufficient. On the other hand, if
  // someone copies the SimpleVector<>, (s)he might expect the data
  // buffer to actually have space for capacity() elements.
  auto data = memoryPool::cuda::make_buffer<SiPixelErrorCompact>(error_h->capacity(), stream, memoryPool::onHost);

  // but transfer only the required amount
  if (not error_h->empty()) {
    cudaCheck(memoryPool::cuda::copy(data, data_d, error_h->size(), stream));
  }
  auto err = *error_h;
  err.set_data(data.get());
  return HostDataError(err, std::move(data));
}



#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
