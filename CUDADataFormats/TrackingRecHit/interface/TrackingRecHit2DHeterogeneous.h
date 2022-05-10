#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DSOAView.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "HeterogeneousCore/CUDAUtilities/interface/memoryPool.h"

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
  using Buffer = typename memoryPool::Buffer<T>;

  using PhiBinner = TrackingRecHit2DSOAView::PhiBinner;

  TrackingRecHit2DHeterogeneous() = default;

  TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                bool isPhase2,
                                int32_t offsetBPIX2,
                                pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                uint32_t const* hitsModuleStart,
                                memoryPool::Where where,
                                cudaStream_t stream,
                                TrackingRecHit2DHeterogeneous const* input = nullptr);

  // used on CPU only
  TrackingRecHit2DHeterogeneous(float* store32,
                                uint16_t* store16,
                                uint32_t* modules,
                                int nHits,
                                memoryPool::Where where = memoryPool::onCPU,
                                cudaStream_t stream = nullptr);
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

  Buffer<float> localCoordToHostAsync(cudaStream_t stream) const;

  Buffer<uint32_t> hitsModuleStartToHostAsync(cudaStream_t stream) const;

  Buffer<uint16_t> store16ToHostAsync(cudaStream_t stream) const;
  Buffer<float> store32ToHostAsync(cudaStream_t stream) const;

  // needed for Host
  void copyFromGPU(TrackingRecHit2DHeterogeneous const* input, cudaStream_t stream);

private:
  static constexpr uint32_t n16 = 4;                 // number of elements in m_store16
  static constexpr uint32_t n32 = 10;                // number of elements in m_store32
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious
  static_assert(n32 == static_cast<uint32_t>(Storage32::kLayers));
  Buffer<uint16_t> m_store16;  //!
  Buffer<float> m_store32;     //!

  Buffer<TrackingRecHit2DSOAView::PhiBinner> m_PhiBinnerStore;              //!
  Buffer<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;  //!

  Buffer<TrackingRecHit2DSOAView> m_view;  //!

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

using TrackingRecHit2DGPU = TrackingRecHit2DHeterogeneous;
using TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous;
using TrackingRecHit2DHost = TrackingRecHit2DHeterogeneous;

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
