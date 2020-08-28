#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include "DataFormats/SiPixelDigi/interface/PixelErrors.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"

#include <cuda_runtime.h>

class SiPixelDigiErrorsCUDA {
public:
  SiPixelDigiErrorsCUDA() = default;
  explicit SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cudaStream_t stream);
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  cms::cuda::SimpleVector<PixelErrorCompact>* error() { return error_d.get(); }
  cms::cuda::SimpleVector<PixelErrorCompact> const* error() const { return error_d.get(); }
  cms::cuda::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.get(); }

  using HostDataError =
      std::pair<cms::cuda::SimpleVector<PixelErrorCompact>, cms::cuda::host::unique_ptr<PixelErrorCompact[]>>;
  HostDataError dataErrorToHostAsync(cudaStream_t stream) const;

  void copyErrorToHostAsync(cudaStream_t stream);

private:
  cms::cuda::device::unique_ptr<PixelErrorCompact[]> data_d;
  cms::cuda::device::unique_ptr<cms::cuda::SimpleVector<PixelErrorCompact>> error_d;
  cms::cuda::host::unique_ptr<cms::cuda::SimpleVector<PixelErrorCompact>> error_h;
  PixelFormatterErrors formatterErrors_h;
};

#endif
