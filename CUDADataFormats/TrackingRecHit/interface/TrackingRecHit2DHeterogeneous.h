#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DSOAView.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

namespace {
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
}  // namespace

template <typename Traits, typename TrackerTraits>
class TrackingRecHit2DHeterogeneousT {
public:
  template <typename>
  friend class TrackingRecHit2DHostT;

  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using TrackingRecHit2DSOAView = TrackingRecHit2DSOAViewT<TrackerTraits>;
  using PhiBinner = typename TrackingRecHit2DSOAView::PhiBinner;
  using AverageGeometry = typename TrackingRecHit2DSOAView::AverageGeometry;

  TrackingRecHit2DHeterogeneousT() = default;

  explicit TrackingRecHit2DHeterogeneousT(uint32_t nHits,
                                          int32_t offsetBPIX2,
                                          pixelCPEforGPU::ParamsOnGPUT<TrackerTraits> const* cpeParams,
                                          uint32_t const* hitsModuleStart,
                                          cudaStream_t stream = nullptr);

  explicit TrackingRecHit2DHeterogeneousT(cms::cuda::host::unique_ptr<float[]>& store32,
                                          cms::cuda::host::unique_ptr<uint16_t[]>& store16,
                                          uint32_t* modules,
                                          int nHits,
                                          cudaStream_t stream = nullptr);
  ~TrackingRecHit2DHeterogeneousT() = default;

  TrackingRecHit2DHeterogeneousT(const TrackingRecHit2DHeterogeneousT&) = delete;
  TrackingRecHit2DHeterogeneousT& operator=(const TrackingRecHit2DHeterogeneousT&) = delete;
  TrackingRecHit2DHeterogeneousT(TrackingRecHit2DHeterogeneousT&&) = default;
  TrackingRecHit2DHeterogeneousT& operator=(TrackingRecHit2DHeterogeneousT&&) = default;

  TrackingRecHit2DSOAView* view() { return m_view.get(); }
  TrackingRecHit2DSOAView const* view() const { return m_view.get(); }

  auto nHits() const { return m_nHits; }
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

protected:
  static constexpr uint32_t n16 = 4;                 // number of elements in m_store16
  static constexpr uint32_t n32 = 10;                // number of elements in m_store32
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious
  static_assert(n32 == static_cast<uint32_t>(Storage32::kLayers));
  unique_ptr<uint16_t[]> m_store16;  //!
  unique_ptr<float[]> m_store32;     //!

  unique_ptr<PhiBinner> m_PhiBinnerStore;              //!
  unique_ptr<AverageGeometry> m_AverageGeometryStore;  //!

  unique_ptr<TrackingRecHit2DSOAView> m_view;  //!

  uint32_t m_nHits;
  int32_t m_offsetBPIX2;

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  PhiBinner* m_phiBinner;
  typename PhiBinner::index_type* m_phiBinnerStorage;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;
};

//Inherit and overload only what we need to overload, remember to use this->
//GPU
template <typename TrackerTraits>
class TrackingRecHit2DGPUT : public TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits, TrackerTraits> {
public:
  using TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits, TrackerTraits>::TrackingRecHit2DHeterogeneousT;

  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint16_t[]> store16ToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<float[]> store32ToHostAsync(cudaStream_t stream) const;
};

//CPU
template <typename TrackerTraits>
class TrackingRecHit2DCPUT : public TrackingRecHit2DHeterogeneousT<cms::cudacompat::CPUTraits, TrackerTraits> {
public:
  using TrackingRecHit2DHeterogeneousT<cms::cudacompat::CPUTraits, TrackerTraits>::TrackingRecHit2DHeterogeneousT;

  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint16_t[]> store16ToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<float[]> store32ToHostAsync(cudaStream_t stream) const;
};

//HOST
template <typename TrackerTraits>
class TrackingRecHit2DHostT : public TrackingRecHit2DHeterogeneousT<cms::cudacompat::HostTraits, TrackerTraits> {
public:
  ~TrackingRecHit2DHostT() = default;
  TrackingRecHit2DHostT() = default;

  explicit TrackingRecHit2DHostT(uint32_t nHits,
                                 int32_t offsetBPIX2,
                                 pixelCPEforGPU::ParamsOnGPUT<TrackerTraits> const* cpeParams,
                                 uint32_t const* hitsModuleStart,
                                 cudaStream_t stream = nullptr)
      : TrackingRecHit2DHeterogeneousT<cms::cudacompat::HostTraits, TrackerTraits>(
            nHits, offsetBPIX2, cpeParams, hitsModuleStart, stream) {}

  explicit TrackingRecHit2DHostT(cms::cuda::host::unique_ptr<float[]>& store32,
                                 cms::cuda::host::unique_ptr<uint16_t[]>& store16,
                                 uint32_t* modules,
                                 int nHits,
                                 cudaStream_t stream = nullptr)
      : TrackingRecHit2DHeterogeneousT<cms::cudacompat::HostTraits, TrackerTraits>(
            store32, store16, modules, nHits, stream) {}

  explicit TrackingRecHit2DHostT(uint32_t nHits,
                                 int32_t offsetBPIX2,
                                 pixelCPEforGPU::ParamsOnGPUT<TrackerTraits> const* cpeParams,
                                 uint32_t const* hitsModuleStart,
                                 cudaStream_t stream,
                                 TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits, TrackerTraits> const* input);
};

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template <typename Traits, typename TrackerTraits>
TrackingRecHit2DHeterogeneousT<Traits, TrackerTraits>::TrackingRecHit2DHeterogeneousT(
    uint32_t nHits,
    int32_t offsetBPIX2,
    pixelCPEforGPU::ParamsOnGPUT<TrackerTraits> const* cpeParams,
    uint32_t const* hitsModuleStart,
    cudaStream_t stream)
    : m_nHits(nHits), m_offsetBPIX2(offsetBPIX2), m_hitsModuleStart(hitsModuleStart) {
  using TrackingRecHit2DSOAView = TrackingRecHit2DSOAViewT<TrackerTraits>;

  auto view = Traits::template make_host_unique<TrackingRecHit2DSOAView>(stream);

  view->m_nHits = nHits;
  m_view = Traits::template make_unique<TrackingRecHit2DSOAView>(stream);  // leave it on host and pass it by value?
  m_AverageGeometryStore = Traits::template make_unique<typename TrackingRecHit2DSOAView::AverageGeometry>(stream);
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

  m_store16 = Traits::template make_unique<uint16_t[]>(nHits * n16, stream);
  m_store32 = Traits::template make_unique<float[]>(nHits * n32 + TrackerTraits::numberOfLayers + 1, stream);
  m_PhiBinnerStore = Traits::template make_unique<typename TrackingRecHit2DSOAView::PhiBinner>(stream);

  static_assert(sizeof(typename TrackingRecHit2DSOAView::hindex_type) == sizeof(float));
  static_assert(sizeof(typename TrackingRecHit2DSOAView::hindex_type) ==
                sizeof(typename TrackingRecHit2DSOAView::PhiBinner::index_type));

  auto get32 = [&](Storage32 i) { return m_store32.get() + static_cast<int>(i) * nHits; };

  // copy all the pointers
  m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
  m_phiBinnerStorage = view->m_phiBinnerStorage =
      reinterpret_cast<typename TrackingRecHit2DSOAView::PhiBinner::index_type*>(get32(Storage32::kPhiStorage));

  view->m_xl = get32(Storage32::kXLocal);
  view->m_yl = get32(Storage32::kYLocal);
  view->m_xerr = get32(Storage32::kXerror);
  view->m_yerr = get32(Storage32::kYerror);
  view->m_chargeAndStatus = reinterpret_cast<uint32_t*>(get32(Storage32::kCharge));

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

  // transfer view
  if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
    cms::cuda::copyAsync(m_view, view, stream);
  } else {
    m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
  }
}

template <typename TrackerTraits>
TrackingRecHit2DHostT<TrackerTraits>::TrackingRecHit2DHostT(
    uint32_t nHits,
    int32_t offsetBPIX2,
    pixelCPEforGPU::ParamsOnGPUT<TrackerTraits> const* cpeParams,
    uint32_t const* hitsModuleStart,
    cudaStream_t stream,
    TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits, TrackerTraits> const* input) {
  using TrackingRecHit2DSOAView = TrackingRecHit2DSOAViewT<TrackerTraits>;

  this->m_nHits = nHits;
  this->m_offsetBPIX2 = offsetBPIX2;
  this->m_hitsModuleStart = hitsModuleStart;

  auto view = cms::cuda::make_host_unique<TrackingRecHit2DSOAView>(stream);

  view->m_nHits = nHits;
  this->m_view =
      cms::cuda::make_host_unique<TrackingRecHit2DSOAView>(stream);  // leave it on host and pass it by value?
  this->m_AverageGeometryStore = cms::cuda::make_host_unique<typename TrackingRecHit2DSOAView::AverageGeometry>(stream);
  view->m_averageGeometry = this->m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    this->m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
    return;
  }

  this->m_store32 = cms::cuda::make_host_unique<float[]>(5 * input->nHits(), stream);
  cms::cuda::copyAsync(this->m_store32, input->m_store32, 5 * input->nHits(), stream);

  static_assert(sizeof(typename TrackingRecHit2DSOAView::hindex_type) == sizeof(float));
  static_assert(sizeof(typename TrackingRecHit2DSOAView::hindex_type) ==
                sizeof(typename TrackingRecHit2DSOAView::PhiBinner::index_type));

  auto get32 = [&](Storage32 i) { return this->m_store32.get() + static_cast<int>(i) * nHits; };

  // copy all the pointers
  this->m_phiBinner = view->m_phiBinner = this->m_PhiBinnerStore.get();
  this->m_phiBinnerStorage = view->m_phiBinnerStorage =
      reinterpret_cast<typename TrackingRecHit2DSOAView::PhiBinner::index_type*>(get32(Storage32::kPhiStorage));

  view->m_xl = get32(Storage32::kXLocal);
  view->m_yl = get32(Storage32::kYLocal);
  view->m_xerr = get32(Storage32::kXerror);
  view->m_yerr = get32(Storage32::kYerror);
  view->m_chargeAndStatus = reinterpret_cast<uint32_t*>(get32(Storage32::kCharge));

  this->m_view = std::move(view);
}

//this is intended to be used only for CPU SoA but doesn't hurt to have it for all cases
template <typename Traits, typename TrackerTraits>
TrackingRecHit2DHeterogeneousT<Traits, TrackerTraits>::TrackingRecHit2DHeterogeneousT(
    cms::cuda::host::unique_ptr<float[]>& store32,
    cms::cuda::host::unique_ptr<uint16_t[]>& store16,
    uint32_t* modules,
    int nHits,
    cudaStream_t stream)
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
  m_PhiBinnerStore = Traits::template make_unique<typename TrackingRecHit2DSOAView::PhiBinner>(stream);
  m_AverageGeometryStore = Traits::template make_unique<typename TrackingRecHit2DSOAView::AverageGeometry>(stream);

  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_hitsModuleStart = m_hitsModuleStart;

  //store transfer
  if constexpr (std::is_same_v<Traits, cms::cudacompat::GPUTraits>) {
    cms::cuda::copyAsync(m_store16, store16, static_cast<int>(n16 * nHits), stream);
    cms::cuda::copyAsync(m_store32, store32, static_cast<int>(n32 * nHits), stream);

  } else {
    std::copy(store32.get(), store32.get() + nHits * n32, m_store32.get());  // want to copy it
    std::copy(store16.get(), store16.get() + nHits * n16, m_store16.get());
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
      reinterpret_cast<typename TrackingRecHit2DSOAView::PhiBinner::index_type*>(get32(Storage32::kPhiStorage));

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

//Classes definition for Phase1/Phase2, to make the classes_def lighter. Not actually used in the code.
using TrackingRecHit2DGPUPhase1 = TrackingRecHit2DGPUT<pixelTopology::Phase1>;
using TrackingRecHit2DCPUPhase1 = TrackingRecHit2DCPUT<pixelTopology::Phase1>;
using TrackingRecHit2DHostPhase1 = TrackingRecHit2DHostT<pixelTopology::Phase1>;

using TrackingRecHit2DGPUPhase2 = TrackingRecHit2DGPUT<pixelTopology::Phase2>;
using TrackingRecHit2DCPUPhase2 = TrackingRecHit2DCPUT<pixelTopology::Phase2>;
using TrackingRecHit2DHostPhase2 = TrackingRecHit2DHostT<pixelTopology::Phase2>;

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneousT_h
