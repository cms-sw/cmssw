#ifndef DataFormats_VertexSoA_interface_OfflineVertexHostCollection_h
#define DataFormats_VertexSoA_interface_OfflineVertexHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/VertexSoA/interface/OfflineVertexSoA.h"

// SoA with x, y, z, id fields in host memory
using OfflineVertexHostCollection = PortableHostCollection<OfflineVertexSoA>;

#endif  // DataFormats_VertexSoA_interface_OfflineVertexHostCollection_h
