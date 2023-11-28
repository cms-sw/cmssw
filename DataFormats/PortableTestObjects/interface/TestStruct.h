#ifndef DataFormats_PortableTestObjects_interface_TestStruct_h
#define DataFormats_PortableTestObjects_interface_TestStruct_h

#include <cstdint>

namespace portabletest {

  // struct with x, y, z, id fields
  struct TestStruct {
    double x;
    double y;
    double z;
    int32_t id;
  };

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_TestStruct_h
