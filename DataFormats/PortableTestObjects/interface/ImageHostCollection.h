#ifndef DataFormats_PortableTestObjects_interface_ImageHostCollection_h
#define DataFormats_PortableTestObjects_interface_ImageHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ImageSoA.h"

namespace portabletest {

  using ImageHostCollection = PortableHostCollection<ImageSoA>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_ImageHostCollection_h
