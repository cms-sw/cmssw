#ifndef DataFormats_PortableTestObjects_interface_TestHostCollection_h
#define DataFormats_PortableTestObjects_interface_TestHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"

namespace portabletest {

  // SoA with x, y, z, id fields in host memory
  using TestHostCollection = PortableHostCollection<TestSoA>;

  using TestHostMultiCollection2 = PortableHostCollection2<TestSoA, TestSoA2>;

  using TestHostMultiCollection3 = PortableHostCollection3<TestSoA, TestSoA2, TestSoA3>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_TestHostCollection_h
