#ifndef DataFormats_HeterogeneousTutorial_interface_TripletsSoA_h
#define DataFormats_HeterogeneousTutorial_interface_TripletsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace tutorial {

  constexpr const int kEmpty = -1;

  GENERATE_SOA_LAYOUT(TripletsSoALayout,
                      // scalars: one value per collection
                      SOA_SCALAR(int, size),
                      // columns: one value per element
                      SOA_COLUMN(int, first),
                      SOA_COLUMN(int, second),
                      SOA_COLUMN(int, third))

  using TripletsSoA = TripletsSoALayout<>;

}  // namespace tutorial

#endif  // DataFormats_HeterogeneousTutorial_interface_TripletsSoA_h
