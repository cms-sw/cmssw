#ifndef DataFormats_VertexGNNReco_interface_VertexGNNHostCollection_h
#define DataFormats_VertexGNNReco_interface_VertexGNNHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/VertexGNNReco/interface/VertexGNNSoA.h"

namespace vertexgnn {

  using TrackFeaturesHostCollection = PortableHostCollection<TrackFeaturesSoA>;
  using GNNOutputHostCollection = PortableHostCollection<GNNOutputSoA>;

}  // namespace vertexgnn

#endif
