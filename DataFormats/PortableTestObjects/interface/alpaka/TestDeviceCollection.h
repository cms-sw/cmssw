#ifndef DataFormats_PortableTestObjects_interface_alpaka_TestDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_TestDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace portabletest {

    // make the names from the top-level portabletest namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portabletest namespace
    using namespace ::portabletest;

    // SoA with x, y, z, id fields in host memory
    using ::portabletest::TestHostCollection;

    // SoA with x, y, z, id fields in device global memory
    using TestDeviceCollection = PortableCollection<TestSoA>;

  }  // namespace portabletest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// check that the portable device collection for the host device is the same as the portable host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::TestDeviceCollection, portabletest::TestHostCollection);

#endif  // DataFormats_PortableTestObjects_interface_alpaka_TestDeviceCollection_h
