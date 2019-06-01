#include "DataFormats/Math/interface/LorentzVector.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/StraightTrajectory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/HelixTrajectory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ForwardSimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/BarrelSimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"

fastsim::Trajectory::~Trajectory() {}

fastsim::Trajectory::Trajectory(const fastsim::Particle &particle) {
  position_ = particle.position();
  momentum_ = particle.momentum();
}

std::unique_ptr<fastsim::Trajectory> fastsim::Trajectory::createTrajectory(const fastsim::Particle &particle,
                                                                           double magneticFieldZ) {
  if (particle.charge() == 0. || magneticFieldZ == 0.) {
    LogDebug("FastSim") << "create straight trajectory";
    return std::unique_ptr<fastsim::Trajectory>(new fastsim::StraightTrajectory(particle));
  } else if (std::abs(particle.momentum().Pt() /
                      (fastsim::Constants::speedOfLight * 1e-4 * particle.charge() * magneticFieldZ)) > 1e5) {
    LogDebug("FastSim") << "create straight trajectory (huge radius)";
    return std::unique_ptr<fastsim::Trajectory>(new fastsim::StraightTrajectory(particle));
  } else {
    LogDebug("FastSim") << "create helix trajectory";
    return std::unique_ptr<fastsim::Trajectory>(new fastsim::HelixTrajectory(particle, magneticFieldZ));
  }
}

double fastsim::Trajectory::nextCrossingTimeC(const fastsim::SimplifiedGeometry &layer, bool onLayer) const {
  if (layer.isForward()) {
    return this->nextCrossingTimeC(static_cast<const fastsim::ForwardSimplifiedGeometry &>(layer), onLayer);
  } else {
    return this->nextCrossingTimeC(static_cast<const fastsim::BarrelSimplifiedGeometry &>(layer), onLayer);
  }
}

double fastsim::Trajectory::nextCrossingTimeC(const fastsim::ForwardSimplifiedGeometry &layer, bool onLayer) const {
  if (onLayer) {
    return -1;
  }
  // t = (z - z_0) / v_z
  // substitute: v_z = p_z / E * c  ( note: extra * c absorbed in p_z units)
  // => t*c = (z - z_0) / p_z * E
  double deltaTimeC = (layer.getZ() - position_.Z()) / momentum_.Z() * momentum_.E();
  return deltaTimeC > 0. ? deltaTimeC : -1.;
}
