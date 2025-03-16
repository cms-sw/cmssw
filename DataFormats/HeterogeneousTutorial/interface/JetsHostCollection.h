#ifndef DataFormats_HeterogeneousTutorial_interface_JetsHostCollection_h
#define DataFormats_HeterogeneousTutorial_interface_JetsHostCollection_h

#include "DataFormats/HeterogeneousTutorial/interface/JetsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace tutorial {

  using JetsHostCollection = PortableHostCollection<JetsSoA>;

}  // namespace tutorial

#endif  // DataFormats_HeterogeneousTutorial_interface_JetsHostCollection_h
