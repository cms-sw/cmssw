#ifndef DataFormats_OfflineVertexSoA_interface_TrackHostCollection_h
#define DataFormats_OfflineVertexSoA_interface_TrackHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/OfflineVertexSoA/interface/TrackSoA.h"

// SoA with x, y, z, id fields in host memory
using TrackHostCollection = PortableHostCollection<TrackSoA>;

#endif  // DataFormats_OfflineVertexSoA_interface_TrackHostCollection_h
