#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DReducedT_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DReducedT_h

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DSOAView.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"

// a reduced (in content and therefore in size) version to be used on CPU for Legacy reconstruction
template <typename TrackerTraits>
class TrackingRecHit2DReducedT {
  using TrackingRecHit2DSOAView = TrackingRecHit2DSOAViewT<TrackerTraits>;

public:
  using HLPstorage = HostProduct<float[]>;
  using HIDstorage = HostProduct<uint16_t[]>;

  template <typename UP32, typename UP16>
  TrackingRecHit2DReducedT(UP32&& istore32, UP16&& istore16, int nhits)
      : m_store32(std::move(istore32)), m_store16(std::move(istore16)), m_nHits(nhits) {
    auto get32 = [&](int i) { return const_cast<float*>(m_store32.get()) + i * nhits; };

    // copy all the pointers (better be in sync with the producer store)

    m_view.m_xl = get32(0);
    m_view.m_yl = get32(1);
    m_view.m_xerr = get32(2);
    m_view.m_yerr = get32(3);
    m_view.m_chargeAndStatus = reinterpret_cast<uint32_t*>(get32(4));
    m_view.m_detInd = const_cast<uint16_t*>(m_store16.get());
  }

  // view only!
  TrackingRecHit2DReducedT(TrackingRecHit2DSOAView const& iview, int nhits) : m_view(iview), m_nHits(nhits) {}

  TrackingRecHit2DReducedT() = default;
  ~TrackingRecHit2DReducedT() = default;

  TrackingRecHit2DReducedT(const TrackingRecHit2DReducedT&) = delete;
  TrackingRecHit2DReducedT& operator=(const TrackingRecHit2DReducedT&) = delete;
  TrackingRecHit2DReducedT(TrackingRecHit2DReducedT&&) = default;
  TrackingRecHit2DReducedT& operator=(TrackingRecHit2DReducedT&&) = default;

  TrackingRecHit2DSOAView& view() { return m_view; }
  TrackingRecHit2DSOAView const& view() const { return m_view; }

  auto nHits() const { return m_nHits; }

private:
  TrackingRecHit2DSOAView m_view;

  HLPstorage m_store32;
  HIDstorage m_store16;

  int m_nHits;
};

using TrackingRecHit2DReducedPhase1 = TrackingRecHit2DReducedT<pixelTopology::Phase1>;
using TrackingRecHit2DReducedPhase2 = TrackingRecHit2DReducedT<pixelTopology::Phase2>;

#endif
