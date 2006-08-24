#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"

ParticleTable*
ParticleTable::myself=0; 

ParticleTable* 
ParticleTable::instance(const DefaultConfig::ParticleDataTable* pdt) {
  if (!myself) myself = new ParticleTable(pdt);
  return myself;
}

ParticleTable* 
ParticleTable::instance() {
  return myself;
}
