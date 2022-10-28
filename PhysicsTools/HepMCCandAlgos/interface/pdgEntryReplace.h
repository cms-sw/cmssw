#ifndef PhysicsTools_HepMCCandAlgos_pdgEntryReplace_h
#define PhysicsTools_HepMCCandAlgos_pdgEntryReplace_h
#include <string>

namespace HepPDT {
  class ParticleDataTable;
}

std::string pdgEntryReplace(const std::string&, HepPDT::ParticleDataTable const&);

#endif
