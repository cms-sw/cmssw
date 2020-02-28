#include "DataFormats/PatCandidates/interface/Vertexing.h"

using pat::VertexAssociation;

void VertexAssociation::setDistances(const AlgebraicVector3 &dist, const AlgebraicSymMatrix33 &err) {
  setDz(Measurement1DFloat(std::abs(dist[2]), std::sqrt(err(2, 2))));

  AlgebraicVector3 dist2D(dist[0], dist[1], 0);
  float d2 = dist2D[0] * dist2D[0] + dist2D[1] * dist2D[1];
  setDr(Measurement1DFloat(sqrt(d2), sqrt(ROOT::Math::Similarity(dist2D, err) / d2)));
}
