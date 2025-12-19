#ifndef DataFormats_PortableTestObjects_interface_TestHostCollection_h
#define DataFormats_PortableTestObjects_interface_TestHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"

namespace portabletest {

  // SoA with x, y, z, id fields in host memory
  using TestHostCollection = PortableHostCollection<TestSoA>;

  using TestHostCollection2 = PortableHostCollection<TestSoABlocks2>;

  using TestHostCollection3 = PortableHostCollection<TestSoABlocks3>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_TestHostCollection_h
