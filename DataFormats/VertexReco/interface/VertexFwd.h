#ifndef VertexReco_VertexFwd_h
#define VertexReco_VertexFwd_h
#include <vector>
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefProd.h"
#include "FWCore/EDProduct/interface/RefVector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ext_collection.h"

namespace reco {
  class Vertex;
  class VertexRefProds {
  public:
    VertexRefProds() { }
    const TracksRef & tracks() const { return tracks_; }
    void setTracks( const TracksRef & ref ) { tracks_ = ref; }
  private:
    TracksRef tracks_;
  };
  typedef ext_collection<std::vector<Vertex>, VertexRefProds> VertexCollection;
  typedef edm::Ref<VertexCollection> VertexRef;
  typedef edm::RefVector<VertexCollection> VertexRefs;
  typedef VertexRefs::iterator vertex_iterator;
}

#endif
