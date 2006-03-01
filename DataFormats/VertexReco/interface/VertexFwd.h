#ifndef VertexReco_VertexFwd_h
#define VertexReco_VertexFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Vertex;
  /// a collection of Vertex objects
  typedef std::vector<Vertex> VertexCollection;
  /// a persistent reference to a Vertex
  typedef edm::Ref<VertexCollection> VertexRef;
  /// a vector of references to Vertex objects in the same collection
  typedef edm::RefVector<VertexCollection> VertexRefs;
  /// iterator over a vector of references to Vertex objects in the same collection
  typedef VertexRefs::iterator vertex_iterator;
}

#endif
