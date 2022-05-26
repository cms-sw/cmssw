#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DSOAView.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

TrackingRecHit2DHeterogeneous::TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                                             bool isPhase2,
                                                             int32_t offsetBPIX2,
                                                             pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                             uint32_t const* hitsModuleStart,
                                                             memoryPool::Where where,
                                                             cudaStream_t stream,
                                                             TrackingRecHit2DHeterogeneous const* input)
    : m_nHits(nHits), m_offsetBPIX2(offsetBPIX2), m_hitsModuleStart(hitsModuleStart) {
  using namespace memoryPool::cuda;

  memoryPool::Deleter deleter = memoryPool::Deleter(std::make_shared<memoryPool::cuda::BundleDelete>(stream, where));
  assert(deleter.pool());
  auto view = makeBuffer<TrackingRecHit2DSOAView>(
      1, stream, memoryPool::onCPU == where ? memoryPool::onCPU : memoryPool::onHost);
  assert(view.get());
  m_nMaxModules = isPhase2 ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;
  assert(view.get());
  view->m_nHits = nHits;
  view->m_nMaxModules = m_nMaxModules;
  m_view = makeBuffer<TrackingRecHit2DSOAView>(
      1, deleter);  // stream, where); // deleter);  // leave it on host and pass it by value?
  assert(m_view.get());
  m_AverageGeometryStore = makeBuffer<TrackingRecHit2DSOAView::AverageGeometry>(1, deleter);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    if (memoryPool::onDevice == where) {
      cudaCheck(memoryPool::cuda::copy(m_view, view, sizeof(TrackingRecHit2DSOAView), stream));
    } else {
      m_view.reset(view.release());
    }
    return;
  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated

  // host copy is "reduced"  (to be reviewed at some point)
  if (memoryPool::onHost == where) {
    // it has to compile for ALL cases
    copyFromGPU(input, stream);
  } else {
    assert(input == nullptr);

    auto nL = isPhase2 ? phase2PixelTopology::numberOfLayers : phase1PixelTopology::numberOfLayers;

    m_store16 = makeBuffer<uint16_t>(nHits * n16, deleter);
    m_store32 = makeBuffer<float>(nHits * n32 + nL + 1, deleter);
    m_PhiBinnerStore = makeBuffer<TrackingRecHit2DSOAView::PhiBinner>(1, deleter);
  }

  static_assert(sizeof(TrackingRecHit2DSOAView::hindex_type) == sizeof(float));
  static_assert(sizeof(TrackingRecHit2DSOAView::hindex_type) == sizeof(TrackingRecHit2DSOAView::PhiBinner::index_type));

  auto get32 = [&](Storage32 i) { return m_store32.get() + static_cast<int>(i) * nHits; };

  // copy all the pointers
  m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
  m_phiBinnerStorage = view->m_phiBinnerStorage =
      reinterpret_cast<TrackingRecHit2DSOAView::PhiBinner::index_type*>(get32(Storage32::kPhiStorage));

  view->m_xl = get32(Storage32::kXLocal);
  view->m_yl = get32(Storage32::kYLocal);
  view->m_xerr = get32(Storage32::kXerror);
  view->m_yerr = get32(Storage32::kYerror);
  view->m_chargeAndStatus = reinterpret_cast<uint32_t*>(get32(Storage32::kCharge));

  if (memoryPool::onHost != where) {
    assert(input == nullptr);
    view->m_xg = get32(Storage32::kXGlobal);
    view->m_yg = get32(Storage32::kYGlobal);
    view->m_zg = get32(Storage32::kZGlobal);
    view->m_rg = get32(Storage32::kRGlobal);

    auto get16 = [&](Storage16 i) { return m_store16.get() + static_cast<int>(i) * nHits; };
    m_iphi = view->m_iphi = reinterpret_cast<int16_t*>(get16(Storage16::kPhi));

    view->m_xsize = reinterpret_cast<int16_t*>(get16(Storage16::kXSize));
    view->m_ysize = reinterpret_cast<int16_t*>(get16(Storage16::kYSize));
    view->m_detInd = get16(Storage16::kDetId);

    m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
    m_hitsLayerStart = view->m_hitsLayerStart = reinterpret_cast<uint32_t*>(get32(Storage32::kLayers));
  }

  // transfer view
  if (memoryPool::onDevice == where) {
    cudaCheck(
        cudaMemcpyAsync(m_view.get(), view.get(), sizeof(TrackingRecHit2DSOAView), cudaMemcpyHostToDevice, stream));
    //    cudaCheck(memoryPool::cuda::copy(m_view, view, sizeof(TrackingRecHit2DSOAView), stream));
  } else {
    m_view.reset(view.release());
  }
}

//this is intended to be used only for CPU SoA but doesn't hurt to have it for all cases
TrackingRecHit2DHeterogeneous::TrackingRecHit2DHeterogeneous(
    float* store32, uint16_t* store16, uint32_t* modules, int nHits, memoryPool::Where where, cudaStream_t stream)
    : m_nHits(nHits), m_hitsModuleStart(modules) {
  using namespace memoryPool::cuda;
  auto view = makeBuffer<TrackingRecHit2DSOAView>(
      1, stream, memoryPool::onCPU == where ? memoryPool::onCPU : memoryPool::onHost);

  m_view = makeBuffer<TrackingRecHit2DSOAView>(1, stream, where);

  view->m_nHits = nHits;

  if (0 == nHits) {
    if (memoryPool::onDevice == where) {
      cudaCheck(memoryPool::cuda::copy(m_view, view, sizeof(TrackingRecHit2DSOAView), stream));
    } else {
      m_view = std::move(view);
    }
    return;
  }

  m_store16 = makeBuffer<uint16_t>(nHits * n16, stream, where);
  m_store32 = makeBuffer<float>(nHits * n32, stream, where);
  m_PhiBinnerStore = makeBuffer<TrackingRecHit2DSOAView::PhiBinner>(1, stream, where);
  m_AverageGeometryStore = makeBuffer<TrackingRecHit2DSOAView::AverageGeometry>(1, stream, where);

  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_hitsModuleStart = m_hitsModuleStart;

  //store transfer
  if (memoryPool::onDevice == where) {
    cudaCheck(cudaMemcpyAsync(m_store32.get(), store32, nHits * n32 * sizeof(float), cudaMemcpyHostToDevice, stream));
    cudaCheck(
        cudaMemcpyAsync(m_store16.get(), store16, nHits * n16 * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
  } else {
    std::copy(store32, store32 + nHits * n32, m_store32.get());  // want to copy it
    std::copy(store16, store16 + nHits * n16, m_store16.get());
  }

  //getters
  auto get32 = [&](Storage32 i) { return m_store32.get() + static_cast<int>(i) * nHits; };
  auto get16 = [&](Storage16 i) { return m_store16.get() + static_cast<int>(i) * nHits; };

  //Store 32
  view->m_xl = get32(Storage32::kXLocal);
  view->m_yl = get32(Storage32::kYLocal);
  view->m_xerr = get32(Storage32::kXerror);
  view->m_yerr = get32(Storage32::kYerror);
  view->m_chargeAndStatus = reinterpret_cast<uint32_t*>(get32(Storage32::kCharge));
  view->m_xg = get32(Storage32::kXGlobal);
  view->m_yg = get32(Storage32::kYGlobal);
  view->m_zg = get32(Storage32::kZGlobal);
  view->m_rg = get32(Storage32::kRGlobal);

  m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
  m_phiBinnerStorage = view->m_phiBinnerStorage =
      reinterpret_cast<TrackingRecHit2DSOAView::PhiBinner::index_type*>(get32(Storage32::kPhiStorage));

  //Store 16
  view->m_detInd = get16(Storage16::kDetId);
  m_iphi = view->m_iphi = reinterpret_cast<int16_t*>(get16(Storage16::kPhi));
  view->m_xsize = reinterpret_cast<int16_t*>(get16(Storage16::kXSize));
  view->m_ysize = reinterpret_cast<int16_t*>(get16(Storage16::kYSize));

  // transfer view
  if (memoryPool::onDevice == where) {
    cudaCheck(memoryPool::cuda::copy(m_view, view, sizeof(TrackingRecHit2DSOAView), stream));
  } else {
    m_view = std::move(view);
  }
}

using namespace memoryPool::cuda;

memoryPool::Buffer<float> TrackingRecHit2DGPU::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = makeBuffer<float>(5 * nHits(), stream, memoryPool::onHost);
  cudaCheck(cudaMemcpyAsync(ret.get(), m_store32.get(), 5 * sizeof(float) * nHits(), cudaMemcpyDeviceToHost, stream));
  return ret;
}

memoryPool::Buffer<float> TrackingRecHit2DGPU::store32ToHostAsync(cudaStream_t stream) const {
  auto ret = makeBuffer<float>(static_cast<int>(n32) * nHits(), stream, memoryPool::onHost);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_store32.get(), static_cast<int>(n32) * sizeof(float) * nHits(), cudaMemcpyDeviceToHost, stream));
  return ret;
}

memoryPool::Buffer<uint16_t> TrackingRecHit2DGPU::store16ToHostAsync(cudaStream_t stream) const {
  auto ret = makeBuffer<uint16_t>(static_cast<int>(n16) * nHits(), stream, memoryPool::onHost);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_store16.get(), static_cast<int>(n16) * sizeof(uint16_t) * nHits(), cudaMemcpyDeviceToHost, stream));
  return ret;
}

memoryPool::Buffer<uint32_t> TrackingRecHit2DGPU::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = makeBuffer<uint32_t>(nMaxModules() + 1, stream, memoryPool::onHost);
  if (m_hitsModuleStart)
    cudaCheck(cudaMemcpyAsync(
        ret.get(), m_hitsModuleStart, sizeof(uint32_t) * (nMaxModules() + 1), cudaMemcpyDeviceToHost, stream));
  return ret;
}

void TrackingRecHit2DHost::copyFromGPU(TrackingRecHit2DGPU const* input, cudaStream_t stream) {
  assert(input);
  m_store32 = input->localCoordToHostAsync(stream);
}
