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
  SiPixelDigiErrorsCUDA(size_t maxFedWords, SiPixelFormatterErrors errors, cudaStream_t stream);
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const SiPixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  SiPixelErrorCompactVector* error() { return error_d.get(); }
  SiPixelErrorCompactVector const* error() const { return error_d.get(); }

  using HostDataError = std::pair<SiPixelErrorCompactVector, memoryPool::buffer<SiPixelErrorCompact>>;
  HostDataError dataErrorToHostAsync(cudaStream_t stream) const;

  void copyErrorToHostAsync(cudaStream_t stream);
  int nErrorWords() const { return nErrorWords_; }

private:
  memoryPool::buffer<SiPixelErrorCompact> data_d;
  memoryPool::buffer<SiPixelErrorCompactVector> error_d;
  memoryPool::buffer<SiPixelErrorCompactVector> error_h;
  SiPixelFormatterErrors formatterErrors_h;
  int nErrorWords_ = 0;
};

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
