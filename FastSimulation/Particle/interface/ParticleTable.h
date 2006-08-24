#ifndef FastSimulation_Particle_ParticleTable_H
#define FastSimulation_Particle_ParticleTable_H

// CLHEP header
#include <CLHEP/HepPDT/DefaultConfig.hh>

class ParticleDataTable;

class ParticleTable {

public:

  /// Get the pointer to the particle data table
  const DefaultConfig::ParticleDataTable* theTable() const {return pdt_;}

  static ParticleTable* instance(const DefaultConfig::ParticleDataTable* pdt);
  static ParticleTable* instance() ;

private:

  ParticleTable(const DefaultConfig::ParticleDataTable* pdt) : pdt_(pdt) {;}
  static ParticleTable* myself;
  const DefaultConfig::ParticleDataTable * pdt_;

};

#endif // FastSimulation_Particle_ParticleTable_H
