#ifndef DataFormats_VertexSoA_interface_VertexGNNHostCollection_h
#define DataFormats_VertexSoA_interface_VertexGNNHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/VertexSoA/interface/VertexGNNSoA.h"

namespace vertexgnn {

  using TrackFeaturesHostCollection = PortableHostCollection<TrackFeaturesSoA>;
  using GNNOutputHostCollection = PortableHostCollection<GNNOutputSoA>;
  using GNNSlotHostCollection = PortableHostCollection<GNNSlotSoA>;

}  // namespace vertexgnn

#endif
