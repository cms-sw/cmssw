#ifndef FastSimulation_Particle_pdg_functions_h
#define FastSimulation_Particle_pdg_functions_h
// -*- C++ -*-
//
// Package:     FastSimulation/Particle
// Class  :     pdg_functions
//
/**\class pdg_functions pdg_functions.h "FastSimulation/Particle/interface/pdg_functions.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 04 Mar 2019 19:49:58 GMT
//

// system include files
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// user include files

// forward declarations

namespace pdg {
  constexpr static double kInvalidMass = -99999;
  double mass(int pdgID, const HepPDT::ParticleDataTable* pdt);

  constexpr static double kInvalidCtau = 1E99;
  double cTau(int pdgID, const HepPDT::ParticleDataTable* pdt);
}  // namespace pdg

#endif
