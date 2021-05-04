
#include "L1Trigger/VertexFinder/interface/Vertex.h"

namespace l1tVertexFinder {

  void Vertex::computeParameters() {
    pT_ = 0.;
    z0_ = 0.;
    float z0square = 0.;
    for (TP track : tracks_) {
      pT_ += track->pt();
      z0_ += track->z0();
      z0square += track->z0() * track->z0();
    }
    z0_ /= tracks_.size();
    z0square /= tracks_.size();
    z0width_ = sqrt(std::abs(z0_ * z0_ - z0square));
  }

}  // end namespace l1tVertexFinder
