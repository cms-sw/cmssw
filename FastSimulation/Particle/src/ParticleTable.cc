#include "FastSimulation/Particle/interface/ParticleTable.h"

thread_local std::unique_ptr<const ParticleTable> 
ParticleTable::myself((const ParticleTable*)NULL); 

void ParticleTable::set(const HepPDT::ParticleDataTable* pdt) { 
  myself.reset( new ParticleTable(pdt) );  
}
