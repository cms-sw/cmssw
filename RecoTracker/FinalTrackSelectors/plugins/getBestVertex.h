#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include <limits>

  using Point = math::XYZPoint;

  inline
  Point getBestVertex(reco::Track const & trk, reco::VertexCollection const & vertices, const size_t minNtracks = 2) {

    Point p_dz(0,0,-99999);
    float dzmin = std::numeric_limits<float>::max();
    for(auto const & vertex : vertices) {
      size_t tracks = vertex.tracksSize();
      if (tracks < minNtracks) {
      continue;
    }
      float dz = std::abs(trk.dz(vertex.position()));
      if(dz < dzmin){
        p_dz = vertex.position();
        dzmin = dz;
      }
    }

    return p_dz;

  }

  inline
  Point getBestVertex_withError(reco::Track const & trk, reco::VertexCollection const & vertices, Point& error, const size_t minNtracks = 2) {

    Point p_dz(0,0,-99999);
    float dzmin = std::numeric_limits<float>::max();
    for(auto const & vertex : vertices) {
      size_t tracks = vertex.tracksSize();
      if (tracks < minNtracks) {
      continue;
    }
      float dz = std::abs(trk.dz(vertex.position()));
      if(dz < dzmin){
        p_dz = vertex.position();
	error.SetXYZ(vertex.xError(),vertex.yError(),vertex.zError());
        dzmin = dz;
      }
    }

    return p_dz;

  }
