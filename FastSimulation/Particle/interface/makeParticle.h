#ifndef FastSimulation_Particle_makeParticle_h
#define FastSimulation_Particle_makeParticle_h
// -*- C++ -*-
//
// Package:     FastSimulation/Particle
// Class  :     makeParticle
//
/**\class makeParticle makeParticle.h "FastSimulation/Particle/interface/makeParticle.h"

 Description: functions to create RawParticle from PDG ids

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 04 Mar 2019 17:15:34 GMT
//

// system include files
#include "DataFormats/Math/interface/LorentzVector.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// user include files

// forward declarations
class RawParticle;

RawParticle makeParticle(HepPDT::ParticleDataTable const*, int id, const math::XYZTLorentzVector& p);
RawParticle makeParticle(HepPDT::ParticleDataTable const*,
                         int id,
                         const math::XYZTLorentzVector& p,
                         const math::XYZTLorentzVector& xStart);

#endif
