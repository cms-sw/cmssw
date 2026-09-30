#ifndef DataFormats_TrackingRecHitsMaskingSoA_interface_alpaka_TrackingRecHitsMaskingSoACollection_h
#define DataFormats_TrackingRecHitsMaskingSoA_interface_alpaka_TrackingRecHitsMaskingSoACollection_h

#include <cstdint>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using TrackingRecHitsMaskingSoACollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                                                 ::reco::TrackingRecHitsMaskingHost,
                                                                 ::reco::TrackingRecHitsMaskingDevice<Device>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::TrackingRecHitsMaskingSoACollection, reco::TrackingRecHitsMaskingHost);

#endif  // DataFormats_TrackingRecHitsMaskingSoA_interface_alpaka_TrackingRecHitsMaskingSoACollection_h
