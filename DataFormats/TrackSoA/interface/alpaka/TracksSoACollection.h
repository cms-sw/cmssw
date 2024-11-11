#ifndef DataFormats_TrackSoA_interface_alpaka_TracksSoACollection_h
#define DataFormats_TrackSoA_interface_alpaka_TracksSoACollection_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

    using ::reco::TracksHost;
    using ::reco::TracksDevice;
    using TracksSoACollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, TracksHost, TracksDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::TracksSoACollection, reco::TracksHost);

#endif  // DataFormats_TrackSoA_interface_alpaka_TracksSoACollection_h
