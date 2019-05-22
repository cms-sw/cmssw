#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"

std::ostream& fastsim::operator<<(std::ostream& os, const fastsim::Particle& particle) {
  os << "fastsim::Particle "
     << " pdgId=" << particle.pdgId_ << " position=(" << particle.position_.X() << "," << particle.position_.Y()
     << " [R="
     << sqrt(particle.position_.X() * particle.position_.X() + particle.position_.Y() * particle.position_.Y()) << "],"
     << particle.position_.Z() << "," << particle.position_.T() << ")"
     << " momentum=(" << particle.momentum_.X() << "," << particle.momentum_.Y() << "," << particle.momentum_.Z() << ","
     << particle.momentum_.T() << ")"
     << " isStable=(" << particle.isStable() << ")"
     << " remainingProperLifeTimeC=" << particle.remainingProperLifeTimeC_;
  return os;
}
