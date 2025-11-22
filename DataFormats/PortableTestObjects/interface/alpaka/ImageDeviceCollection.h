#ifndef DataFormats_PortableTestObjects_interface_alpaka_ImageDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_ImageDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/ImageHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ImageSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace portabletest {

    // make the names from the top-level portabletest namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portabletest namespace
    using namespace ::portabletest;

    using ImageDeviceCollection = PortableCollection<ImageSoA>;

  }  // namespace portabletest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// heterogeneous ml data checks
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::ImageDeviceCollection, portabletest::ImageHostCollection);

#endif  // DataFormats_PortableTestObjects_interface_alpaka_ImageDeviceCollection_h
