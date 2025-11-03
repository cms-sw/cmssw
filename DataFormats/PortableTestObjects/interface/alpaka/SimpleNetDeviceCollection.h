#ifndef DataFormats_PortableTestObjects_interface_alpaka_SimpleNetDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_SimpleNetDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/SimpleNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SimpleNetSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace portabletest {

    // make the names from the top-level portabletest namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portabletest namespace
    using namespace ::portabletest;

    using SimpleNetDeviceCollection = PortableCollection<SimpleNetSoA>;

  }  // namespace portabletest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// heterogeneous ml data checks
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::SimpleNetDeviceCollection, portabletest::SimpleNetHostCollection);

#endif  // DataFormats_PortableTestObjects_interface_alpaka_SimpleNetDeviceCollection_h
