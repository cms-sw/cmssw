#ifndef FastSimulation_Particle_ParticleTable_H
#define FastSimulation_Particle_ParticleTable_H

// HepPDT header
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class ParticleTable {

public:
  struct Sentry{
    Sentry(const HepPDT::ParticleDataTable* pdt) {       
      ParticleTable::instance()->set(pdt); 
    }
    ~Sentry() {       
      ParticleTable::instance()->set(nullptr); 
    }
  };

  ~ParticleTable() {     
  }

  /// Get the pointer to the particle data table
  const HepPDT::ParticleDataTable* theTable() const {    
    return pdt_; 
  }

  static ParticleTable* const instance() {     
    if( !myself ) myself = new ParticleTable();
    return myself; 
  }

private:
  
  ParticleTable(const HepPDT::ParticleDataTable* pdt=nullptr) : pdt_(pdt) {}
  void set( const HepPDT::ParticleDataTable* pdt) { pdt_ = pdt; } 
  static thread_local ParticleTable* myself;

  const HepPDT::ParticleDataTable* pdt_;

  friend struct Sentry;
};



#endif // FastSimulation_Particle_ParticleTable_H
