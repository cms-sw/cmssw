#ifndef RecoBTag_SecondaryVertex_SecondaryVertex_h
#define RecoBTag_SecondaryVertex_SecondaryVertex_h

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoBTag/SecondaryVertex/interface/TemplatedSecondaryVertex.h"

namespace reco {

typedef TemplatedSecondaryVertex<reco::Vertex> SecondaryVertex; 

} // namespace reco

#endif // RecoBTag_SecondaryVertex_SecondaryVertex_h
