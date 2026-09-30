#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskingHost_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskingHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingSoA.h"

namespace reco {

  using TrackingRecHitsMaskingHost = PortableHostCollection<TrackingRecHitsMaskingSoA>;

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskingHost_h
