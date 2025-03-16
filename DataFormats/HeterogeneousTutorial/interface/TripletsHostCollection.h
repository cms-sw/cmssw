#ifndef DataFormats_HeterogeneousTutorial_interface_TripletsHostCollection_h
#define DataFormats_HeterogeneousTutorial_interface_TripletsHostCollection_h

#include "DataFormats/HeterogeneousTutorial/interface/TripletsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace tutorial {

  using TripletsHostCollection = PortableHostCollection<TripletsSoA>;

}  // namespace tutorial

#endif  // DataFormats_HeterogeneousTutorial_interface_TripletsHostCollection_h
