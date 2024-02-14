#ifndef DataFormats_PortableTestObjects_interface_TestHostObject_h
#define DataFormats_PortableTestObjects_interface_TestHostObject_h

#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/PortableTestObjects/interface/TestStruct.h"

namespace portabletest {

  // struct with x, y, z, id fields in host memory
  using TestHostObject = PortableHostObject<TestStruct>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_TestHostObject_h
