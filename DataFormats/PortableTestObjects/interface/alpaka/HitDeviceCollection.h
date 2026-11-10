#ifndef DataFormats_PortableTestObjects_interface_alpaka_HitDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_HitDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/HitHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/HitSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace portabletest {

    // make the names from the top-level portabletest namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portabletest namespace
    using namespace ::portabletest;

    using HitDeviceCollection = PortableCollection<HitSoA>;
    using HitToTrackDeviceCollection = PortableCollection<HitToTrackSoA>;
    using TrackBeginDeviceCollection = PortableCollection<TrackBeginSoA>;
  }  // namespace portabletest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// heterogeneous ml data checks
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::HitDeviceCollection, portabletest::HitHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::HitToTrackDeviceCollection, portabletest::HitToTrackHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::TrackBeginDeviceCollection, portabletest::TrackBeginHostCollection);

#endif  // DataFormats_PortableTestObjects_interface_alpaka_HitDeviceCollection_h
