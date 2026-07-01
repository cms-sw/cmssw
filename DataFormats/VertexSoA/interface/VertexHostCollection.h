#ifndef DataFormats_VertexSoA_interface_VertexHostCollection_h
#define DataFormats_VertexSoA_interface_VertexHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/VertexSoA/interface/VertexSoA.h"

// SoA with x, y, z, id fields in host memory
using VertexHostCollection = PortableHostCollection<VertexSoA>;

#endif  // DataFormats_VertexSoA_interface_VertexHostCollection_h
