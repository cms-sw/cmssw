#ifndef DataFormats_PortableTestObjects_interface_TestHostCollection_h
#define DataFormats_PortableTestObjects_interface_TestHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"

namespace portabletest {

  // SoA with x, y, z, id fields in host memory
  using TestHostCollection = PortableHostCollection<TestSoA>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_TestHostCollection_h
