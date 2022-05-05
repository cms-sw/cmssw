#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DSOAView.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

template <typename Traits>
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
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using PhiBinner = TrackingRecHit2DSOAView::PhiBinner;

  TrackingRecHit2DHeterogeneous() = default;

  explicit TrackingRecHit2DHeterogeneous(
      uint32_t nHits,
      bool isPhase2,
      int32_t offsetBPIX2,
      pixelCPEforGPU::ParamsOnGPU const* cpeParams,
      uint32_t const* hitsModuleStart,
      cudaStream_t stream,
      TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits> const* input = nullptr);

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

  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const;

  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;

  cms::cuda::host::unique_ptr<uint16_t[]> store16ToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<float[]> store32ToHostAsync(cudaStream_t stream) const;

  // needs specialization for Host
  void copyFromGPU(TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits> const* input, cudaStream_t stream);

private:
  static constexpr uint32_t n16 = 4;                 // number of elements in m_store16
  static constexpr uint32_t n32 = 10;                // number of elements in m_store32
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious
  static_assert(n32 == static_cast<uint32_t>(Storage32::kLayers));
  unique_ptr<uint16_t[]> m_store16;  //!
  unique_ptr<float[]> m_store32;     //!

  unique_ptr<TrackingRecHit2DSOAView::PhiBinner> m_PhiBinnerStore;              //!
  unique_ptr<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;  //!

  unique_ptr<TrackingRecHit2DSOAView> m_view;  //!

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

using TrackingRecHit2DGPU = TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits>;
using TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous<cms::cudacompat::CPUTraits>;
using TrackingRecHit2DHost = TrackingRecHit2DHeterogeneous<cms::cudacompat::HostTraits>;

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template <typename Traits>
TrackingRecHit2DHeterogeneous<Traits>::TrackingRecHit2DHeterogeneous(
    uint32_t nHits,
    bool isPhase2,
    int32_t offsetBPIX2,
    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
    uint32_t const* hitsModuleStart,
    cudaStream_t stream,
    TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits> const* input)
    : m_nHits(nHits), m_offsetBPIX2(offsetBPIX2), m_hitsModuleStart(hitsModuleStart) {
  auto view = Traits::template make_host_unique<TrackingRecHit2DSOAView>(stream);

  m_nMaxModules = isPhase2 ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;

  view->m_nHits = nHits;
  view->m_nMaxModules = m_nMaxModules;
  m_view = Traits::template make_unique<TrackingRecHit2DSOAView>(stream);  // leave it on host and pass it by value?
  m_AverageGeometryStore = Traits::template make_unique<TrackingRecHit2DSOAView::AverageGeometry>(stream);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
      cms::cuda::copyAsync(m_view, view, stream);
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
  if constexpr (std::is_same_v<Traits, cms::cudacompat::HostTraits>) {
    // it has to compile for ALL cases
    copyFromGPU(input, stream);
  } else {
    assert(input == nullptr);

    auto nL = isPhase2 ? phase2PixelTopology::numberOfLayers : phase1PixelTopology::numberOfLayers;

    m_store16 = Traits::template make_unique<uint16_t[]>(nHits * n16, stream);
    m_store32 = Traits::template make_unique<float[]>(nHits * n32 + nL + 1, stream);
    m_PhiBinnerStore = Traits::template make_unique<TrackingRecHit2DSOAView::PhiBinner>(stream);
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

  if constexpr (!std::is_same_v<Traits, cms::cudacompat::HostTraits>) {
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
  if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
    cms::cuda::copyAsync(m_view, view, stream);
  } else {
    m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
  }
}

//this is intended to be used only for CPU SoA but doesn't hurt to have it for all cases
template <typename Traits>
TrackingRecHit2DHeterogeneous<Traits>::TrackingRecHit2DHeterogeneous(
    float* store32, uint16_t* store16, uint32_t* modules, int nHits, cudaStream_t stream)
    : m_nHits(nHits), m_hitsModuleStart(modules) {
  auto view = Traits::template make_host_unique<TrackingRecHit2DSOAView>(stream);

  m_view = Traits::template make_unique<TrackingRecHit2DSOAView>(stream);

  view->m_nHits = nHits;

  if (0 == nHits) {
    if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
      cms::cuda::copyAsync(m_view, view, stream);
    } else {
      m_view = std::move(view);
    }
    return;
  }

  m_store16 = Traits::template make_unique<uint16_t[]>(nHits * n16, stream);
  m_store32 = Traits::template make_unique<float[]>(nHits * n32, stream);
  m_PhiBinnerStore = Traits::template make_unique<TrackingRecHit2DSOAView::PhiBinner>(stream);
  m_AverageGeometryStore = Traits::template make_unique<TrackingRecHit2DSOAView::AverageGeometry>(stream);

  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_hitsModuleStart = m_hitsModuleStart;

  //store transfer
  if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
    cms::cuda::copyAsync(m_store16, store16, stream);
    cms::cuda::copyAsync(m_store32, store32, stream);
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
  if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
    cms::cuda::copyAsync(m_view, view, stream);
  } else {
    m_view = std::move(view);
  }
}

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
