#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"

#include <iostream>
#include <TH1F.h>

std::ostream& fastsim::operator<<(std::ostream& os, const SimplifiedGeometry& layer) {
  os << (layer.isForward() ? "ForwardSimplifiedGeometry" : "BarrelSimplifiedGeometry") << " index=" << layer.index_
     << (layer.isForward() ? " z=" : " radius=") << layer.geomProperty_;
  return os;
}

// note: define destructor and constructor in .cc file,
//       otherwise one cannot forward declare TH1F in the header file
//       w/o compilation issues
fastsim::SimplifiedGeometry::~SimplifiedGeometry() {}

fastsim::SimplifiedGeometry::SimplifiedGeometry(double geomProperty)
    : geomProperty_(geomProperty),
      index_(-1),
      detLayer_(nullptr),
      nuclearInteractionThicknessFactor_(1.),
      caloType_(fastsim::SimplifiedGeometry::NONE) {}
