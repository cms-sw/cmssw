#include <cassert>

// #include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
    : m_store(memoryPool::cuda::makeBuffer<SiPixelDigisCUDA::StoreType>(
          SiPixelDigisCUDASOAView::roundFor128ByteAlignment(maxFedWords) *
              static_cast<int>(SiPixelDigisCUDASOAView::StorageLocation::kMAX),
          stream,
          memoryPool::onDevice)),
      m_view(m_store, maxFedWords, SiPixelDigisCUDASOAView::StorageLocation::kMAX) {
  assert(maxFedWords != 0);
}

memoryPool::Buffer<SiPixelDigisCUDA::StoreType> SiPixelDigisCUDA::copyAllToHostAsync(cudaStream_t stream) const {
  auto ret = memoryPool::cuda::makeBuffer<StoreType>(
      m_view.roundFor128ByteAlignment(nDigis()) * static_cast<int>(SiPixelDigisCUDASOAView::StorageLocationHost::kMAX),
      stream,
      memoryPool::onHost);
  cudaCheck(cudaMemcpyAsync(ret.get(),
                            m_view.clus(),
                            m_view.roundFor128ByteAlignment(nDigis()) * sizeof(SiPixelDigisCUDA::StoreType) *
                                static_cast<int>(SiPixelDigisCUDASOAView::StorageLocationHost::kMAX),
                            cudaMemcpyDeviceToHost,
                            stream));
  return ret;
}
