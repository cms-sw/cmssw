#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
    : m_store(cms::cuda::make_device_unique<uint16_t[]>(
          maxFedWords * int(SiPixelDigisCUDASOAView::StorageLocation::MAX), stream)),
      m_view(cms::cuda::make_host_unique<SiPixelDigisCUDASOAView>(stream)) {
  auto get16 = [&](SiPixelDigisCUDASOAView::StorageLocation s) { return m_store.get() + int(s) * maxFedWords; };
  m_view->xx_ = get16(SiPixelDigisCUDASOAView::StorageLocation::XX);
  m_view->yy_ = get16(SiPixelDigisCUDASOAView::StorageLocation::YY);
  m_view->adc_ = get16(SiPixelDigisCUDASOAView::StorageLocation::ADC);
  m_view->moduleInd_ = get16(SiPixelDigisCUDASOAView::StorageLocation::MODULEIND);
  m_view->clus_ = reinterpret_cast<int32_t*>(get16(SiPixelDigisCUDASOAView::StorageLocation::CLUS));
  m_view->pdigi_ = reinterpret_cast<uint32_t*>(get16(SiPixelDigisCUDASOAView::StorageLocation::PDIGI));
  m_view->rawIdArr_ = reinterpret_cast<uint32_t*>(get16(SiPixelDigisCUDASOAView::StorageLocation::RAWIDARR));
}

cms::cuda::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::copyAllToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis() * int(SiPixelDigisCUDASOAView::StorageLocationHost::MAX),
                                                     stream);
  cudaCheck(cudaMemcpyAsync(ret.get(),
                            view()->clus_,
                            nDigis() * sizeof(uint16_t) * int(SiPixelDigisCUDASOAView::StorageLocationHost::MAX),
                            cudaMemcpyDeviceToHost,
                            stream));
  return ret;
}
