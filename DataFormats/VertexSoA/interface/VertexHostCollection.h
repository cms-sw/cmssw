#ifndef DataFormats_VertexSoA_interface_VertexHostCollection_h
#define DataFormats_VertexSoA_interface_VertexHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/VertexSoA/interface/VertexSoA.h"

namespace reco {
    using VertexHostCollection = PortableHostCollection<VertexSoA>;
}
#endif  // DataFormats_VertexSoA_interface_VertexHostCollection_h
