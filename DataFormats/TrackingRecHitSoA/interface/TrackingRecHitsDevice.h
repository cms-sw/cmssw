#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitSoADevice_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitSoADevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TrackerTraits, typename TDev>
class TrackingRecHitDevice : public PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>, TDev> {
public:
  using hitSoA = TrackingRecHitSoA<TrackerTraits>;
  //Need to decorate the class with the inherited portable accessors being now a template
  using PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>, TDev>::view;
  using PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>, TDev>::const_view;
  using PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>, TDev>::buffer;

  TrackingRecHitDevice() = default;

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackingRecHitDevice(uint32_t nHits, int32_t offsetBPIX2, uint32_t const* hitsModuleStart, TQueue queue)
      : PortableDeviceCollection<TrackingRecHitLayout<TrackerTraits>, TDev>(nHits, queue) {
    const auto device = alpaka::getDev(queue);

    auto start_h = cms::alpakatools::make_host_view(hitsModuleStart, TrackerTraits::numberOfModules + 1);
    auto start_d =
        cms::alpakatools::make_device_view(device, view().hitsModuleStart().data(), TrackerTraits::numberOfModules + 1);
    alpaka::memcpy(queue, start_d, start_h);

    auto off_h = cms::alpakatools::make_host_view(offsetBPIX2);
    auto off_d = cms::alpakatools::make_device_view(device, view().offsetBPIX2());
    alpaka::memcpy(queue, off_d, off_h);
    alpaka::wait(queue);
  }

  uint32_t nHits() const { return view().metadata().size(); }
  uint32_t const* hitsModuleStart() const { return view().hitsModuleStart().data(); }
};
#endif  // DataFormats_RecHits_interface_TrackingRecHitSoADevice_h
