#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TrackerTraits>
class TrackingRecHitHost : public PortableHostCollection<TrackingRecHitLayout<TrackerTraits>> {
public:
  using hitSoA = TrackingRecHitSoA<TrackerTraits>;

  // Need to decorate the class with the inherited portable accessors being now a template
  using PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>::view;
  using PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>::const_view;
  using PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>::buffer;

  TrackingRecHitHost() = default;

  // Constructor which specifies only the SoA size, to be used when copying the results from the device to the host
  template <typename TQueue>
  explicit TrackingRecHitHost(TQueue queue, uint32_t nHits)
      : PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>(nHits, queue) {}

  // Constructor which specifies the SoA size, number of BPIX1 hits, and the modules entry points
  template <typename TQueue>
  explicit TrackingRecHitHost(TQueue queue, uint32_t nHits, int32_t offsetBPIX2, uint32_t const* hitsModuleStart)
      : PortableHostCollection<TrackingRecHitLayout<TrackerTraits>>(nHits, queue) {
    std::copy(hitsModuleStart, hitsModuleStart + TrackerTraits::numberOfModules + 1, view().hitsModuleStart().data());
    view().offsetBPIX2() = offsetBPIX2;
  }

  uint32_t nHits() const { return view().metadata().size(); }

  int32_t offsetBPIX2() const { return view().offsetBPIX2(); }

  uint32_t const* hitsModuleStart() const { return view().hitsModuleStart().data(); }

  // do nothing for a host collection
  template <typename TQueue>
  void updateFromDevice(TQueue) {}
};

using TrackingRecHitHostPhase1 = TrackingRecHitHost<pixelTopology::Phase1>;
using TrackingRecHitHostPhase2 = TrackingRecHitHost<pixelTopology::Phase2>;
using TrackingRecHitHostHIonPhase1 = TrackingRecHitHost<pixelTopology::HIonPhase1>;

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h
