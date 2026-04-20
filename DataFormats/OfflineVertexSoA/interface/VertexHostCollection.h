#ifndef DataFormats_OfflineVertexSoA_interface_VertexHostCollection_h
#define DataFormats_OfflineVertexSoA_interface_VertexHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/OfflineVertexSoA/interface/VertexSoA.h"

// SoA with x, y, z, id fields in host memory
using VertexHostCollection = PortableHostCollection<VertexSoA>;

#endif  // DataFormats_OfflineVertexSoA_interface_VertexHostCollection_h
