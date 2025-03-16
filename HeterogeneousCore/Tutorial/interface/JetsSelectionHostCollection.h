#ifndef HeterogeneousCore_Tutorial_interface_JetsSelectionHostCollection_h
#define HeterogeneousCore_Tutorial_interface_JetsSelectionHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/Tutorial/interface/JetsSelectionSoA.h"

namespace tutorial {

  using JetsSelectionHostCollection = PortableHostCollection<JetsSelectionSoA>;

}  // namespace tutorial

#endif  // HeterogeneousCore_Tutorial_interface_JetsSelectionHostCollection_h
