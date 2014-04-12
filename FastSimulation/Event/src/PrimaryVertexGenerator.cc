#include "FastSimulation/Event/interface/PrimaryVertexGenerator.h"

  /// Default constructor
PrimaryVertexGenerator::PrimaryVertexGenerator() : 
  math::XYZVector(),
  boost_(0)
{
}

PrimaryVertexGenerator::~PrimaryVertexGenerator() { 
  if ( boost_ ) delete boost_; 
}

TMatrixD* 
PrimaryVertexGenerator::boost() const { 
  return boost_;
}

void 
PrimaryVertexGenerator::setBoost(TMatrixD* aBoost) {
  boost_ = aBoost;
}
