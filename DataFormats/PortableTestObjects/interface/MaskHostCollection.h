#ifndef DataFormats_PortableTestObjects_interface_MaskHostCollection_h
#define DataFormats_PortableTestObjects_interface_MaskHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MaskSoA.h"

namespace portabletest {

  using MaskHostCollection = PortableHostCollection<MaskSoA>;
  using ScalarMaskHostCollection = PortableHostCollection<ScalarMaskSoA>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_MaskHostCollection_h
