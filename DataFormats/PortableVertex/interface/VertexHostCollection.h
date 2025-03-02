#ifndef DataFormats_PortableVertex_interface_VertexHostCollection_h
#define DataFormats_PortableVertex_interface_VertexHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableVertex/interface/VertexSoA.h"

namespace portablevertex {

  // SoA with x, y, z, id fields in host memory
  using VertexHostCollection = PortableHostCollection<VertexSoA>;
  using TrackHostCollection = PortableHostCollection<TrackSoA>;
  using ClusterParamsHostCollection = PortableHostCollection<ClusterParamsSoA>;
}  // namespace portablevertex

#endif  // DataFormats_PortableVertex_interface_VertexHostCollection_h
