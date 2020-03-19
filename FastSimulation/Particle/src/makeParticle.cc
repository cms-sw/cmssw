// -*- C++ -*-
//
// Package:     FastSimulation/Particle
// Class  :     makeParticle
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 04 Mar 2019 17:15:41 GMT
//

// system include files

// user include files
#include "FastSimulation/Particle/interface/makeParticle.h"
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"

inline RawParticle unchecked_makeParticle(int id, const math::XYZTLorentzVector& p, double mass, double charge) {
  return RawParticle(id, p, mass, charge);
}

inline RawParticle unchecked_makeParticle(
    int id, const math::XYZTLorentzVector& p, const math::XYZTLorentzVector& xStart, double mass, double charge) {
  return RawParticle(id, p, xStart, mass, charge);
}

RawParticle makeParticle(HepPDT::ParticleDataTable const* table, int id, const math::XYZTLorentzVector& p) {
  double charge = 0.;
  double mass = 0.;
  auto info = table->particle(HepPDT::ParticleID(id));
  if (info) {
    charge = info->charge();
    mass = info->mass().value();
  }

  return unchecked_makeParticle(id, p, mass, charge);
}

RawParticle makeParticle(HepPDT::ParticleDataTable const* table,
                         int id,
                         const math::XYZTLorentzVector& p,
                         const math::XYZTLorentzVector& xStart) {
  double charge = 0.;
  double mass = 0.;
  auto info = table->particle(HepPDT::ParticleID(id));
  if (info) {
    charge = info->charge();
    mass = info->mass().value();
  }
  return unchecked_makeParticle(id, p, xStart, mass, charge);
}
