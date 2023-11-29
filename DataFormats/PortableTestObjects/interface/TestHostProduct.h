#ifndef DataFormats_PortableTestObjects_interface_TestHostProduct_h
#define DataFormats_PortableTestObjects_interface_TestHostProduct_h

#include "DataFormats/Portable/interface/PortableHostProduct.h"
#include "DataFormats/PortableTestObjects/interface/TestStruct.h"

namespace portabletest {

  // struct with x, y, z, id fields in host memory
  using TestHostProduct = PortableHostProduct<TestStruct>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_TestHostProduct_h
