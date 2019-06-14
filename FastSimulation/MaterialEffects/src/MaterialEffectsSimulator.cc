
#include "FastSimulation/MaterialEffects/interface/MaterialEffectsSimulator.h"

#include <list>

using std::list;
using std::pair;

MaterialEffectsSimulator::MaterialEffectsSimulator(double A, double Z, double density, double radLen)
    : A(A), Z(Z), density(density), radLen(radLen) {
  _theUpdatedState.clear();
}

MaterialEffectsSimulator::~MaterialEffectsSimulator() {
  // Don't delete the objects contained in the list
  _theUpdatedState.clear();
}

void MaterialEffectsSimulator::updateState(ParticlePropagator& Particle,
                                           double radlen,
                                           RandomEngineAndDistribution const* random) {
  _theUpdatedState.clear();
  theClosestChargedDaughterId = -1;

  radLengths = radlen;
  if (radLengths > 0.)
    compute(Particle, random);
}

XYZVector MaterialEffectsSimulator::orthogonal(const XYZVector& aVector) const {
  double x = fabs(aVector.X());
  double y = fabs(aVector.Y());
  double z = fabs(aVector.Z());

  if (x < y)
    return x < z ? XYZVector(0., aVector.Z(), -aVector.Y()) : XYZVector(aVector.Y(), -aVector.X(), 0.);
  else
    return y < z ? XYZVector(-aVector.Z(), 0., aVector.X()) : XYZVector(aVector.Y(), -aVector.X(), 0.);
}
