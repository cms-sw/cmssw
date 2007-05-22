#ifndef FastSimulation_Particle_ParticleTable_H
#define FastSimulation_Particle_ParticleTable_H

// HepPDT header
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

class ParticleTable {

public:

  /// Get the pointer to the particle data table
  const HepPDT::ParticleDataTable* theTable() const {return pdt_;}

  static ParticleTable* instance(const HepPDT::ParticleDataTable* pdt);
  static inline ParticleTable* instance() { return myself; }

private:

  ParticleTable(const HepPDT::ParticleDataTable* pdt) : pdt_(pdt) {;}
  static ParticleTable* myself;
  const HepPDT::ParticleDataTable * pdt_;

};

#endif // FastSimulation_Particle_ParticleTable_H
