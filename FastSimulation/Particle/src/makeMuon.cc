// -*- C++ -*-
//
// Package:     FastSimulation/Particle
// Class  :     makeMuon
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 04 Mar 2019 17:36:54 GMT
//

// system include files

// user include files
#include "FastSimulation/Particle/interface/makeMuon.h"

#include "FastSimulation/Particle/interface/makeParticle.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
namespace rawparticle {
  RawParticle makeMuon(bool isParticle, const math::XYZTLorentzVector& p, 
                       const math::XYZTLorentzVector& xStart) {
    if(isParticle) {
      return makeParticle(ParticleTable::instance(), 13, p,xStart);
    }
    return makeParticle(ParticleTable::instance(), -13,p,xStart);
  }
}
