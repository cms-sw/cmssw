#ifndef DataFormats_TrackingRecHitsSoA_interface_TrackingRecHitsMaskingDevice_h
#define DataFormats_TrackingRecHitsSoA_interface_TrackingRecHitsMaskingDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingSoA.h"

namespace reco {

  template <typename TDev>
  using TrackingRecHitsMaskingDevice = PortableDeviceCollection<TDev, TrackingRecHitsMaskingSoA>;

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskingDevice_h
