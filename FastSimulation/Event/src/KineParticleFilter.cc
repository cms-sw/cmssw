#include "FastSimulation/Event/interface/KineParticleFilter.h"
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

//#define EDM_ML_DEBUG

KineParticleFilter::KineParticleFilter(const edm::ParameterSet& cfg) {
  // Charged particles must have pt greater than chargedPtMin [GeV]
  double chargedPtMin = cfg.getParameter<double>("chargedPtMin");
  chargedPtMin2 = chargedPtMin * chargedPtMin;

  // Particles must have energy greater than EMin [GeV]
  EMin = cfg.getParameter<double>("EMin");

  // Allow *ALL* protons with energy > protonEMin
  protonEMin = cfg.getParameter<double>("protonEMin");

  // Particles must have abs(eta) < etaMax (if close enough to 0,0,0)
  double etaMax = cfg.getParameter<double>("etaMax");
  cos2ThetaMax = (std::exp(2. * etaMax) - 1.) / (std::exp(2. * etaMax) + 1.);
  cos2ThetaMax *= cos2ThetaMax;

  // Particles must have vertex inside the volume enclosed by ECAL
  double vertexRMax = cfg.getParameter<double>("rMax");
  vertexRMax2 = vertexRMax * vertexRMax;
  vertexZMax = cfg.getParameter<double>("zMax");
}

bool KineParticleFilter::acceptParticle(const RawParticle& particle) const {
  int pId = abs(particle.pid());

  // skipp invisible particles
  if (pId == 12 || pId == 14 || pId == 16 || pId == 1000022) {
#ifdef EDM_ML_DEBUG
    std::cout << "KineParticleFilter: reject particle with pId = " << pId << std::endl;
#endif
    return false;
  }

  // keep all high-energy protons
  else if (pId == 2212 && particle.E() >= protonEMin) {
    bool accepted = acceptVertex(particle.vertex());
#ifdef EDM_ML_DEBUG
    if (!accepted) std::cout << "KineParticleFilter: reject proton with E = " << particle.E() << " > " << protonEMin << " because of vertex" << std::endl;
#endif
    return accepted;
  }

  // cut on the energy
  else if (particle.E() < EMin) {
#ifdef EDM_ML_DEBUG
    std::cout << "KineParticleFilter: reject particle with E = " << particle.E() << " < " << EMin << std::endl;
#endif
    return false;
  }

  // cut on pt of charged particles
  else if (particle.charge() != 0 && particle.Perp2() < chargedPtMin2) {
#ifdef EDM_ML_DEBUG
    std::cout << "KineParticleFilter: reject particle with charge = " << particle.charge() << " and pT = " << particle.pt() << " < " << chargedPtMin2 << std::endl;
#endif
    return false;
  }

  // cut on eta if the origin vertex is close to the beam
  else if (particle.vertex().Perp2() < 25. && particle.cos2Theta() > cos2ThetaMax) {
#ifdef EDM_ML_DEBUG
    std::cout << "KineParticleFilter: reject particle with cos2Theta = " << particle.cos2Theta() << " < " << cos2ThetaMax << std::endl;
#endif
    return false;
  }

  // particles must have vertex in volume enclosed by ECAL
  return acceptVertex(particle.vertex());
}

bool KineParticleFilter::acceptVertex(const XYZTLorentzVector& vertex) const {
  bool accepted = vertex.Perp2() < vertexRMax2 && fabs(vertex.Z()) < vertexZMax;
#ifdef EDM_ML_DEBUG
  if (!accepted) std::cout << "KineParticleFilter: reject particle because of vertex with R = " << vertex.Rho() << " (max " << std::sqrt(vertexRMax2) << ") and Z = " << fabs(vertex.Z()) << " (max " << vertexZMax << ")" << std::endl;
#endif
  return accepted;
}
