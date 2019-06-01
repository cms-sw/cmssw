// -*- C++ -*-
//
// Package:     CommonTools/BaseParticlePropagator
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
#include "CommonTools/BaseParticlePropagator/interface/makeMuon.h"

#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"
namespace rawparticle {
  RawParticle makeMuon(bool isParticle, const math::XYZTLorentzVector& p, const math::XYZTLorentzVector& xStart) {
    constexpr double kMass = 0.10566;  //taken from SimGeneral/HepPDTESSource/data/particle.tbl
    if (isParticle) {
      return RawParticle(13, p, xStart, kMass, -1.);
    }
    return RawParticle(-13, p, xStart, kMass, +1.);
  }
}  // namespace rawparticle
