#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

TrackingRecHit2DCUDA::TrackingRecHit2DCUDA(uint32_t nHits,
                                           pixelCPEforGPU::ParamsOnGPU const *cpeParams,
                                           uint32_t const *hitsModuleStart,
                                           cuda::stream_t<> &stream)
    : m_nHits(nHits), m_hitsModuleStart(hitsModuleStart) {
  edm::Service<CUDAService> cs;

  auto view = cs->make_host_unique<TrackingRecHit2DSOAView>(stream);
  view->m_nHits = nHits;
  m_view = cs->make_device_unique<TrackingRecHit2DSOAView>(stream);

  // if empy do not bother
  if (0 == nHits) {
    cudautils::copyAsync(m_view, view, stream);
    return;
  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated
  m_store16 = cs->make_device_unique<uint16_t[]>(nHits * n16, stream);
  m_store32 = cs->make_device_unique<float[]>(nHits * n32 + 11, stream);
  m_HistStore = cs->make_device_unique<TrackingRecHit2DSOAView::Hist>(stream);

  auto get16 = [&](int i) { return m_store16.get() + i * nHits; };
  auto get32 = [&](int i) { return m_store32.get() + i * nHits; };

  // copy all the pointers
  m_hist = view->m_hist = m_HistStore.get();

  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  view->m_xl = get32(0);
  view->m_yl = get32(1);
  view->m_xerr = get32(2);
  view->m_yerr = get32(3);

  view->m_xg = get32(4);
  view->m_yg = get32(5);
  view->m_zg = get32(6);
  view->m_rg = get32(7);

  m_iphi = view->m_iphi = reinterpret_cast<int16_t *>(get16(0));

  view->m_charge = reinterpret_cast<int32_t *>(get32(8));
  view->m_xsize = reinterpret_cast<int16_t *>(get16(2));
  view->m_ysize = reinterpret_cast<int16_t *>(get16(3));
  view->m_detInd = get16(1);

  m_hitsLayerStart = view->m_hitsLayerStart = reinterpret_cast<uint32_t *>(get32(n32));

  // transfer view
  cudautils::copyAsync(m_view, view, stream);
}

cudautils::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(cuda::stream_t<> &stream) const {
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<float[]>(4 * nHits(), stream);
  cudautils::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

cudautils::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(
    cuda::stream_t<> &stream) const {
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<uint32_t[]>(2001, stream);
  cudaMemcpyAsync(ret.get(), m_hitsModuleStart, 4 * 2001, cudaMemcpyDefault, stream.id());
  return ret;
}
