#include "FastSimulation/SimplifiedGeometryPropagator/interface/ParticleFilter.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "vdt/vdtMath.h"

fastsim::ParticleFilter::ParticleFilter(const edm::ParameterSet& cfg) {
  // Charged particles must have pt greater than chargedPtMin [GeV]
  double chargedPtMin = cfg.getParameter<double>("chargedPtMin");
  chargedPtMin2_ = chargedPtMin * chargedPtMin;

  // (All) particles must have energy greater than EMin [GeV]
  EMin_ = cfg.getParameter<double>("EMin");

  // Allow *ALL* protons with energy > protonEMin
  protonEMin_ = cfg.getParameter<double>("protonEMin");

  // List of invisible particles if extension needed
  // Predefined: Neutrinos, Neutralino_1
  skipParticles_ = cfg.getParameter<std::vector<int>>("invisibleParticles");

  // Particles must have abs(eta) < etaMax (if close enough to 0,0,0)
  double etaMax = cfg.getParameter<double>("etaMax");
  cos2ThetaMax_ = (vdt::fast_exp(2. * etaMax) - 1.) / (vdt::fast_exp(2. * etaMax) + 1.);
  cos2ThetaMax_ *= cos2ThetaMax_;

  // Particles must have vertex inside the tracker
  vertexRMax2_ = 129.0 * 129.0;
  vertexZMax_ = 303.353;
}

bool fastsim::ParticleFilter::accepts(const fastsim::Particle& particle) const {
  int absPdgId = abs(particle.pdgId());

  // skip invisible particles
  if (absPdgId == 12 || absPdgId == 14 || absPdgId == 16 || absPdgId == 1000022) {
    return false;
  }
  // keep all high-energy protons
  else if (absPdgId == 2212 && particle.momentum().E() >= protonEMin_) {
    return true;
  }

  // cut on eta if the origin vertex is close to the beam
  else if (particle.position().Perp2() < 25. &&
           particle.momentum().Pz() * particle.momentum().Pz() / particle.momentum().P2() > cos2ThetaMax_) {
    return false;
  }

  // possible to extend list of invisible particles
  for (unsigned InvIdx = 0; InvIdx < skipParticles_.size(); InvIdx++) {
    if (absPdgId == abs(skipParticles_.at(InvIdx))) {
      return false;
    }
  }

  // particles must have vertex in volume of tracker
  return acceptsVtx(particle.position()) && acceptsEn(particle);
  //return (acceptsVtx(particle.position()) || particle.momentum().Pz()*particle.momentum().Pz()/particle.momentum().P2() > (vdt::fast_exp(2.*3.0)-1.) / (vdt::fast_exp(2.*3.0)+1.)*(vdt::fast_exp(2.*3.0)-1.) / (vdt::fast_exp(2.*3.0)+1.)) && acceptsEn(particle);
}

bool fastsim::ParticleFilter::acceptsEn(const fastsim::Particle& particle) const {
  int absPdgId = abs(particle.pdgId());

  // keep all high-energy protons
  if (absPdgId == 2212 && particle.momentum().E() >= protonEMin_) {
    return true;
  }

  // cut on the energy
  else if (particle.momentum().E() < EMin_) {
    return false;
  }

  // cut on pt of charged particles
  else if (particle.charge() != 0 && particle.momentum().Perp2() < chargedPtMin2_) {
    return false;
  }

  return true;
}

bool fastsim::ParticleFilter::acceptsVtx(const math::XYZTLorentzVector& originVertex) const {
  // origin vertex is within the tracker volume
  return (originVertex.Perp2() < vertexRMax2_ && fabs(originVertex.Z()) < vertexZMax_);
}
