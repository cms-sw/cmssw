#ifndef FastSimulation_Particle_ParticleTable_H
#define FastSimulation_Particle_ParticleTable_H

// HepPDT header
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include <memory>


class ParticleTable {

public:
  struct Sentry{
    Sentry(const HepPDT::ParticleDataTable* pdt) { ParticleTable::set(pdt); }
    ~Sentry() { ParticleTable::set(NULL); }
  };

  /// Get the pointer to the particle data table
  const HepPDT::ParticleDataTable* theTable() const { return pdt_; }

  static const ParticleTable* instance() 
  { return myself.get(); }

private:
 ParticleTable(const HepPDT::ParticleDataTable* pdt) : pdt_(pdt) {}
  static void set( const HepPDT::ParticleDataTable* );
  static thread_local std::unique_ptr<const ParticleTable> myself;

  const HepPDT::ParticleDataTable* pdt_;

  friend struct Sentry;
};

#endif // FastSimulation_Particle_ParticleTable_H
