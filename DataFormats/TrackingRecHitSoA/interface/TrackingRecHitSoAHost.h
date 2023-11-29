#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TrackerTraits>
class TrackingRecHitHost : public PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>> {
public:
  using hitSoA = TrackingRecHitAlpakaSoA<TrackerTraits>;
  //Need to decorate the class with the inherited portable accessors being now a template
  using PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>::view;
  using PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>::const_view;
  using PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>::buffer;

  TrackingRecHitHost() = default;

  template <typename TQueue>
  explicit TrackingRecHitHost(uint32_t nHits, TQueue queue)
      : PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>(nHits, queue) {}

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackingRecHitHost(uint32_t nHits, int32_t offsetBPIX2, uint32_t const* hitsModuleStart, TQueue queue)
      : PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>(nHits, queue) {
    std::copy(hitsModuleStart, hitsModuleStart + TrackerTraits::numberOfModules + 1, view().hitsModuleStart().data());
    view().offsetBPIX2() = offsetBPIX2;
  }

  uint32_t nHits() const { return view().metadata().size(); }
  uint32_t const* hitsModuleStart() const { return view().hitsModuleStart().data(); }
};

using TrackingRecHitHostPhase1 = TrackingRecHitHost<pixelTopology::Phase1>;
using TrackingRecHitHostPhase2 = TrackingRecHitHost<pixelTopology::Phase2>;
using TrackingRecHitHostHIonPhase1 = TrackingRecHitHost<pixelTopology::HIonPhase1>;

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h
