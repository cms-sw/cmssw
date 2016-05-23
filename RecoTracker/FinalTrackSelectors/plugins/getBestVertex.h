#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <limits>

namespace {

  using Point = math::XYZPoint;


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

  /* 
    Point getBestVertex(reco::Track const & trk,, VertexCollection const & vertices)  {

    //    Point p(0,0,-99999);
    Point p_dz(0,0,-99999);
    // float bestWeight = 0;
    float dzmin = 10000;
  for(auto const & vertex : vertices){
    // auto w = vertex.trackWeight(track);
    Point v_pos = vertex.position();
    //    if(w > bestWeight){
    //  p = v_pos;
    //   bestWeight = w;
    //} else if (0 == bestWeight) {
    float dz = std::abs(trK.dz(v_pos));
    if(dz < dzmin){
      p_dz = v_pos;
      dzmin = dz;
    }
    //}
  }

  return (bestWeight > 0) ? p : p_dz;

  }
  */
}
