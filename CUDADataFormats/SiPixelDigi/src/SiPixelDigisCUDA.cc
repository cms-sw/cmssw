#include <cassert>

#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
    : m_store(cms::cuda::make_device_unique<SiPixelDigisCUDA::m_store_type[]>(
          roundFor128ByteAlignment(maxFedWords) * int(SiPixelDigisCUDASOAView::StorageLocation::kMAX), stream)) {
  assert(maxFedWords != 0);
  auto get16 = [&](SiPixelDigisCUDASOAView::StorageLocation s) {
    return m_store.get() + int(s) * roundFor128ByteAlignment(maxFedWords);
  };
  m_view.xx_ = get16(SiPixelDigisCUDASOAView::StorageLocation::kXX);
  m_view.yy_ = get16(SiPixelDigisCUDASOAView::StorageLocation::kYY);
  m_view.adc_ = get16(SiPixelDigisCUDASOAView::StorageLocation::kADC);
  m_view.moduleInd_ = get16(SiPixelDigisCUDASOAView::StorageLocation::kMODULEIND);
  m_view.clus_ = reinterpret_cast<int32_t*>(get16(SiPixelDigisCUDASOAView::StorageLocation::kCLUS));
  m_view.pdigi_ = reinterpret_cast<uint32_t*>(get16(SiPixelDigisCUDASOAView::StorageLocation::kPDIGI));
  m_view.rawIdArr_ = reinterpret_cast<uint32_t*>(get16(SiPixelDigisCUDASOAView::StorageLocation::kRAWIDARR));
}

cms::cuda::host::unique_ptr<SiPixelDigisCUDA::m_store_type[]> SiPixelDigisCUDA::copyAllToHostAsync(
    cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<m_store_type[]>(
      roundFor128ByteAlignment(nDigis()) * int(SiPixelDigisCUDASOAView::StorageLocationHost::kMAX), stream);
  cudaCheck(cudaMemcpyAsync(ret.get(),
                            view().clus(),
                            roundFor128ByteAlignment(nDigis()) * sizeof(SiPixelDigisCUDA::m_store_type) *
                                int(SiPixelDigisCUDASOAView::StorageLocationHost::kMAX),
                            cudaMemcpyDeviceToHost,
                            stream));
  return ret;
}
