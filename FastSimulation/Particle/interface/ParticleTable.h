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
      edm::LogError("particletable") << "  Sentry() - set pdt to: " << pdt 
				     << ' ' << ParticleTable::myself 
				     << std::endl;
      ParticleTable::myself->set(pdt); 
    }
    ~Sentry() { 
      edm::LogError("particletable") << " ~Sentry() - set pdt to NULL" 
				      << ' ' << ParticleTable::myself 
				     << std::endl;
      ParticleTable::myself->set(nullptr); 
    }
  };

  ~ParticleTable() { 
    edm::LogError("particletable") << "~ParticleTable()" << std::endl;
  }

  /// Get the pointer to the particle data table
  const HepPDT::ParticleDataTable* theTable() const { 
    edm::LogError("particletable") << "Asked for theTable at : " << pdt_ << std::endl;
    return pdt_; 
  }

  static ParticleTable* const instance() { 
    edm::LogError("particletable") << "Asked for myself : " << &myself << std::endl;
    return myself; 
  }

private:
  ParticleTable() : pdt_(nullptr) {}
  ParticleTable(const HepPDT::ParticleDataTable* pdt=nullptr) : pdt_(pdt) {}
  void set( const HepPDT::ParticleDataTable* pdt) { pdt_ = pdt; } 
  static thread_local ParticleTable* myself;

  const HepPDT::ParticleDataTable* pdt_;

  friend struct Sentry;
};



#endif // FastSimulation_Particle_ParticleTable_H
