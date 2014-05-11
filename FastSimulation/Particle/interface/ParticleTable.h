#ifndef FastSimulation_Particle_ParticleTable_H
#define FastSimulation_Particle_ParticleTable_H

// HepPDT header
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include <atomic>
#include <mutex>

class ParticleTable {

public:

  /// Get the pointer to the particle data table
  const HepPDT::ParticleDataTable* theTable() const {return pdt_;}

  static ParticleTable* instance(const HepPDT::ParticleDataTable* pdt=NULL);

private:

  ParticleTable(const HepPDT::ParticleDataTable* pdt) : pdt_(pdt) {}
  static std::atomic<ParticleTable*> myself;
  static std::mutex _mutex;
  const HepPDT::ParticleDataTable* pdt_;

};

#endif // FastSimulation_Particle_ParticleTable_H
