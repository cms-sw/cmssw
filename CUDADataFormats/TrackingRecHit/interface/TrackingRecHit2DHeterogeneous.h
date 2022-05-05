#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DSOAView.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

template <memoryPool::Where where>
class TrackingRecHit2DHeterogeneous {
public:
  enum class Storage32 {
    kXLocal = 0,
    kYLocal = 1,
    kXerror = 2,
    kYerror = 3,
    kCharge = 4,
    kXGlobal = 5,
    kYGlobal = 6,
    kZGlobal = 7,
    kRGlobal = 8,
    kPhiStorage = 9,
    kLayers = 10
  };

  enum class Storage16 {
    kDetId = 0,
    kPhi = 1,
    kXSize = 2,
    kYSize = 3,
  };

  template <typename T>
  using buffer = typename memoryPool::buffer<T>;

  using PhiBinner = TrackingRecHit2DSOAView::PhiBinner;

  TrackingRecHit2DHeterogeneous() = default;

  explicit TrackingRecHit2DHeterogeneous(
      uint32_t nHits,
      bool isPhase2,
      int32_t offsetBPIX2,
      pixelCPEforGPU::ParamsOnGPU const* cpeParams,
      uint32_t const* hitsModuleStart,
      cudaStream_t stream,
      TrackingRecHit2DHeterogeneous<memoryPool::onDevice> const* input = nullptr);

  explicit TrackingRecHit2DHeterogeneous(
      float* store32, uint16_t* store16, uint32_t* modules, int nHits, cudaStream_t stream = nullptr);
  ~TrackingRecHit2DHeterogeneous() = default;

  TrackingRecHit2DHeterogeneous(const TrackingRecHit2DHeterogeneous&) = delete;
  TrackingRecHit2DHeterogeneous& operator=(const TrackingRecHit2DHeterogeneous&) = delete;
  TrackingRecHit2DHeterogeneous(TrackingRecHit2DHeterogeneous&&) = default;
  TrackingRecHit2DHeterogeneous& operator=(TrackingRecHit2DHeterogeneous&&) = default;

  TrackingRecHit2DSOAView* view() { return m_view.get(); }
  TrackingRecHit2DSOAView const* view() const { return m_view.get(); }

  auto nHits() const { return m_nHits; }
  auto nMaxModules() const { return m_nMaxModules; }
  auto offsetBPIX2() const { return m_offsetBPIX2; }

  auto hitsModuleStart() const { return m_hitsModuleStart; }
  auto hitsLayerStart() { return m_hitsLayerStart; }
  auto phiBinner() { return m_phiBinner; }
  auto phiBinnerStorage() { return m_phiBinnerStorage; }
  auto iphi() { return m_iphi; }

  buffer<float> localCoordToHostAsync(cudaStream_t stream) const;

  buffer<uint32_t> hitsModuleStartToHostAsync(cudaStream_t stream) const;

  buffer<uint16_t> store16ToHostAsync(cudaStream_t stream) const;
  buffer<float> store32ToHostAsync(cudaStream_t stream) const;

  // needs specialization for Host
  void copyFromGPU(TrackingRecHit2DHeterogeneous<memoryPool::onDevice> const* input, cudaStream_t stream);

private:
  static constexpr uint32_t n16 = 4;                 // number of elements in m_store16
  static constexpr uint32_t n32 = 10;                // number of elements in m_store32
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious
  static_assert(n32 == static_cast<uint32_t>(Storage32::kLayers));
  buffer<uint16_t> m_store16;  //!
  buffer<float> m_store32;     //!

  buffer<TrackingRecHit2DSOAView::PhiBinner> m_PhiBinnerStore;              //!
  buffer<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;  //!

  buffer<TrackingRecHit2DSOAView> m_view;  //!

  uint32_t m_nHits;
  int32_t m_offsetBPIX2;

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  uint32_t m_nMaxModules;
  // needed as kernel params...
  PhiBinner* m_phiBinner;
  PhiBinner::index_type* m_phiBinnerStorage;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;
};

using TrackingRecHit2DGPU = TrackingRecHit2DHeterogeneous<memoryPool::onDevice>;
using TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous<memoryPool::onCPU>;
using TrackingRecHit2DHost = TrackingRecHit2DHeterogeneous<memoryPool::onHost>;


template <memoryPool::Where where>
TrackingRecHit2DHeterogeneous<where>::TrackingRecHit2DHeterogeneous(
    uint32_t nHits,
    bool isPhase2,
    int32_t offsetBPIX2,
    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
    uint32_t const* hitsModuleStart,
    cudaStream_t stream,
    TrackingRecHit2DHeterogeneous<memoryPool::onDevice> const* input)
    : m_nHits(nHits), m_offsetBPIX2(offsetBPIX2), m_hitsModuleStart(hitsModuleStart) {

  using namespace memoryPool::cuda;
  auto view = make_buffer<TrackingRecHit2DSOAView>(1,stream,memoryPool::onCPU==where ? memoryPool::onCPU : memoryPool::onHost);

  m_nMaxModules = isPhase2 ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;

  view->m_nHits = nHits;
  view->m_nMaxModules = m_nMaxModules;
  m_view = make_buffer<TrackingRecHit2DSOAView>(1,stream,where);  // leave it on host and pass it by value?
  m_AverageGeometryStore = make_buffer<TrackingRecHit2DSOAView::AverageGeometry>(1,stream,where);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    if constexpr (memoryPool::onDevice == where) {
      cudaCheck(memoryPool::cuda::copy(m_view, view, sizeof(TrackingRecHit2DSOAView), stream));
    } else {
      m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
    }
    return;
  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated

  // host copy is "reduced"  (to be reviewed at some point)
  if constexpr (memoryPool::onHost == where) {
    // it has to compile for ALL cases
    copyFromGPU(input, stream);
  } else {
    assert(input == nullptr);

    auto nL = isPhase2 ? phase2PixelTopology::numberOfLayers : phase1PixelTopology::numberOfLayers;

    m_store16 = make_buffer<uint16_t>(nHits * n16, stream, where);
    m_store32 = make_buffer<float>(nHits * n32 + nL + 1, stream, where);
    m_PhiBinnerStore = make_buffer<TrackingRecHit2DSOAView::PhiBinner>(1,stream,where);
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

  if constexpr (memoryPool::onHost != where) {
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
  if constexpr (memoryPool::onDevice == where) {
    cudaCheck(memoryPool::cuda::copy(m_view, view, sizeof(TrackingRecHit2DSOAView), stream));
  } else {
    m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
  }
}

//this is intended to be used only for CPU SoA but doesn't hurt to have it for all cases
template <memoryPool::Where where>
TrackingRecHit2DHeterogeneous<where>::TrackingRecHit2DHeterogeneous(
    float* store32, uint16_t* store16, uint32_t* modules, int nHits, cudaStream_t stream)
    : m_nHits(nHits), m_hitsModuleStart(modules) {

  using namespace memoryPool::cuda;
  auto view = make_buffer<TrackingRecHit2DSOAView>(1,stream,memoryPool::onCPU==where? memoryPool::onCPU : memoryPool::onHost);

  m_view = make_buffer<TrackingRecHit2DSOAView>(1,stream,where);

  view->m_nHits = nHits;

  if (0 == nHits) {
    if constexpr (memoryPool::onDevice == where) {
     cudaCheck(memoryPool::cuda::copy(m_view, view, sizeof(TrackingRecHit2DSOAView), stream));
    } else {
      m_view = std::move(view);
    }
    return;
  }

  m_store16 = make_buffer<uint16_t>(nHits * n16, stream,where);
  m_store32 = make_buffer<float>(nHits * n32, stream,where);
  m_PhiBinnerStore = make_buffer<TrackingRecHit2DSOAView::PhiBinner>(1,stream,where);
  m_AverageGeometryStore = make_buffer<TrackingRecHit2DSOAView::AverageGeometry>(1,stream,where);

  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_hitsModuleStart = m_hitsModuleStart;

  //store transfer
  if constexpr (memoryPool::onDevice == where) {
     cudaCheck(cudaMemcpyAsync(m_store32.get(), store32, nHits * n32, cudaMemcpyHostToDevice,stream));
     cudaCheck(cudaMemcpyAsync(m_store16.get(), store16, nHits * n16, cudaMemcpyHostToDevice,stream));
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
  if constexpr (memoryPool::onDevice == where) {
     cudaCheck(memoryPool::cuda::copy(m_view, view, sizeof(TrackingRecHit2DSOAView), stream));
  } else {
      m_view = std::move(view);
  }
}

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
