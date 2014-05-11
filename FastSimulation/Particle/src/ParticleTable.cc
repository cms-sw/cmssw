#include "FastSimulation/Particle/interface/ParticleTable.h"

std::atomic<ParticleTable*> ParticleTable::myself(NULL); 

std::mutex ParticleTable::_mutex;

ParticleTable* 
ParticleTable::instance(const HepPDT::ParticleDataTable* pdt) {  
  ParticleTable* tmp  = myself.load(std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_acquire);
  if( tmp == NULL ) {
    std::lock_guard<std::mutex> lock(_mutex);
    tmp = myself.load(std::memory_order_relaxed);
    if( tmp == NULL && pdt ) { 
      // && means we load the singleton the first time we get non-null pdt
      // since we have acquired a lock here this is thread safe
      tmp = new ParticleTable(pdt);
      std::atomic_thread_fence(std::memory_order_release);
      myself.store(tmp);
    }
  }
  return tmp;
}
