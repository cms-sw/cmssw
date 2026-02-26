#ifndef DataFormats_PortableTestObjects_interface_MultiHeadNetHostCollection_h
#define DataFormats_PortableTestObjects_interface_MultiHeadNetHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MultiHeadNetSoA.h"

namespace portabletest {

  using MultiHeadNetHostCollection = PortableHostCollection<MultiHeadNetSoA>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_MultiHeadNetHostCollection_h
