#ifndef FastSimulation_Particle_makeMuon_h
#define FastSimulation_Particle_makeMuon_h
// -*- C++ -*-
//
// Package:     FastSimulation/Particle
// Class  :     makeMuon
// 
/**\class makeMuon makeMuon.h "FastSimulation/Particle/interface/makeMuon.h"

 Description: Creates a RawParticle of type muon

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 04 Mar 2019 17:36:32 GMT
//

// system include files
#include "DataFormats/Math/interface/LorentzVector.h"

// user include files

// forward declarations
class RawParticle;

namespace rawparticle {
  ///Create a particle with momentum 'p' at space-time point xStart
  /// The particle will be a muon if iParticle==true, else it will
  /// be an anti-muon.
  RawParticle makeMuon(bool isParticle, const math::XYZTLorentzVector& p, 
                       const math::XYZTLorentzVector& xStart);
}

#endif
