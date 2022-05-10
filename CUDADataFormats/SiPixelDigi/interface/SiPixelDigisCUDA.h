#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/memoryPool.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDASOAView.h"

class SiPixelDigisCUDA {
public:
  using StoreType = uint16_t;
  SiPixelDigisCUDA() = default;
  SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream);
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

  /*inline*/ memoryPool::Buffer<StoreType> copyAllToHostAsync(cudaStream_t stream) const;

  SiPixelDigisCUDASOAView view() { return m_view; }
  SiPixelDigisCUDASOAView const view() const { return m_view; }

private:
  // These are consumed by downstream device code
  memoryPool::Buffer<StoreType> m_store;

  SiPixelDigisCUDASOAView m_view;

  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
