#ifndef DataFormats_PortableTestObjects_interface_SimpleNetHostCollection_h
#define DataFormats_PortableTestObjects_interface_SimpleNetHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SimpleNetSoA.h"

namespace portabletest {

  using SimpleNetHostCollection = PortableHostCollection<SimpleNetSoA>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_SimpleNetHostCollection_h
