#ifndef VertexReco_VertexFwd_h
#define VertexReco_VertexFwd_h
#include <vector>
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefVector.h"

namespace reco {
  class Vertex;
  typedef std::vector<Vertex> VertexCollection;
  typedef edm::Ref<VertexCollection> VertexRef;
  typedef edm::RefVector<VertexCollection> VertexRefs;
  typedef VertexRefs::iterator vertex_iterator;
}

#endif
