#include <cassert>

#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
    : m_store(cms::cuda::make_device_unique<SiPixelDigisCUDA::StoreType[]>(
          SiPixelDigisCUDASOAView::roundFor128ByteAlignment(maxFedWords) *
              static_cast<int>(SiPixelDigisCUDASOAView::StorageLocation::kMAX),
          stream)),
      m_view(m_store, maxFedWords, SiPixelDigisCUDASOAView::StorageLocation::kMAX) {
  assert(maxFedWords != 0);
}

cms::cuda::host::unique_ptr<SiPixelDigisCUDA::StoreType[]> SiPixelDigisCUDA::copyAllToHostAsync(
    cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<StoreType[]>(
      m_view.roundFor128ByteAlignment(nDigis()) * static_cast<int>(SiPixelDigisCUDASOAView::StorageLocationHost::kMAX),
      stream);
  cudaCheck(cudaMemcpyAsync(ret.get(),
                            m_view.clus(),
                            m_view.roundFor128ByteAlignment(nDigis()) * sizeof(SiPixelDigisCUDA::StoreType) *
                                static_cast<int>(SiPixelDigisCUDASOAView::StorageLocationHost::kMAX),
                            cudaMemcpyDeviceToHost,
                            stream));
  return ret;
}
