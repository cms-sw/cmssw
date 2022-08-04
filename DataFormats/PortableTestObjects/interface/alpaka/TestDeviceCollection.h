#ifndef DataFormats_PortableTestObjects_interface_alpaka_TestDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_TestDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace portabletest {

    // import the top-level portabletest namespace
    using namespace ::portabletest;

    // SoA with x, y, z, id fields in device global memory
    using TestDeviceCollection = PortableCollection<TestSoA>;

  }  // namespace portabletest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_PortableTestObjects_interface_alpaka_TestDeviceCollection_h
