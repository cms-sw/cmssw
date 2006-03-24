#ifndef VertexReco_VertexFwd_h
#define VertexReco_VertexFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Vertex;
  typedef std::vector<Vertex> VertexCollection;
  typedef edm::Ref<VertexCollection> VertexRef;
  typedef edm::RefVector<VertexCollection> VertexRefs;
  typedef VertexRefs::iterator vertex_iterator;
}

#endif
