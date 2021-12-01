#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDASOAView.h"

class SiPixelDigisCUDA {
public:
  using m_store_type = uint16_t;
  SiPixelDigisCUDA() = default;
  explicit SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream);
  ~SiPixelDigisCUDA() = default;

  SiPixelDigisCUDA(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA &operator=(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA(SiPixelDigisCUDA &&) = default;
  SiPixelDigisCUDA &operator=(SiPixelDigisCUDA &&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

  cms::cuda::host::unique_ptr<m_store_type[]> copyAllToHostAsync(cudaStream_t stream) const;

  SiPixelDigisCUDASOAView view() { return m_view; }
  SiPixelDigisCUDASOAView const view() const { return m_view; }

private:
  // These are consumed by downstream device code
  cms::cuda::device::unique_ptr<m_store_type[]> m_store;

  SiPixelDigisCUDASOAView m_view;  // "me" pointer

  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
