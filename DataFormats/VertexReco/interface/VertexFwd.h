#ifndef VertexReco_VertexFwd_h
#define VertexReco_VertexFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Vertex;
  /// collection of Vertex objects
  typedef std::vector<Vertex> VertexCollection;
  /// persistent reference to a Vertex
  typedef edm::Ref<VertexCollection> VertexRef;
  /// persistent reference to a Vertex
  typedef edm::RefProd<VertexCollection> VertexRefProd;
  /// vector of references to Vertex objects in the same collection
  typedef edm::RefVector<VertexCollection> VertexRefVector;
  /// iterator over a vector of references to Vertex objects in the same collection
  typedef VertexRefVector::iterator vertex_iterator;
  /// persistent reference to a Vertex, using views
  typedef edm::RefToBase<reco::Vertex> VertexBaseRef;
}

#endif
