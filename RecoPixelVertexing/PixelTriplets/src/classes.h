#include "RecoPixelVertexing/PixelTriplets/interface/IntermediateHitTriplets.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>

namespace RecoPixelVertexing_PixelTriplets {
  struct dictionary {
    IntermediateHitTriplets iht;
    edm::Wrapper<IntermediateHitTriplets> wiht;
  };
}
