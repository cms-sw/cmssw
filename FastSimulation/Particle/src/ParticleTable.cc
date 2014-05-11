#include "FastSimulation/Particle/interface/ParticleTable.h"

std::atomic<ParticleTable*> ParticleTable::myself; 

std::mutex ParticleTable::_mutex;

ParticleTable* 
ParticleTable::instance(const HepPDT::ParticleDataTable* pdt) {  
  ParticleTable* tmp  = myself.load(std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_acquire);
  if( tmp == NULL ) {
    std::lock_guard<std::mutex> lock(_mutex);
    tmp = myself.load(std::memory_order_relaxed);
    if( tmp == NULL || pdt ) { // load a new singleton if given data table
      // since we have acquired a lock here this is thread safe
      if( tmp ) delete tmp;
      tmp = new ParticleTable(pdt);
      std::atomic_thread_fence(std::memory_order_release);
      myself.store(tmp);
    }
  }
  return tmp;
}
