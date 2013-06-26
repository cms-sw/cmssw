#include "FastSimulation/Particle/interface/ParticleTable.h"

ParticleTable*
ParticleTable::myself=0; 

ParticleTable* 
ParticleTable::instance(const HepPDT::ParticleDataTable* pdt) {
  if (!myself) myself = new ParticleTable(pdt);
  return myself;
}
