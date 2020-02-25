#ifndef L1TkTrigger_L1TkEtMissParticleFwd_h
#define L1TkTrigger_L1TkEtMissParticleFwd_h
// Package:     L1Trigger
// Class  :     L1TkEtMissParticleFwd

// system include files
// user include files
// forward declarations
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {
   class L1TkEtMissParticle ;
   typedef std::vector< L1TkEtMissParticle > L1TkEtMissParticleCollection ;
   //typedef edm::RefProd< L1TkEtMissParticle > L1TkEtMissParticleRefProd ;
   //typedef edm::Ref< L1TkEtMissParticleCollection > L1TkEtMissParticleRef ;
   //typedef edm::RefVector< L1TkEtMissParticleCollection > L1TkEtMissParticleRefVector ;
   //typedef std::vector< L1TkEtMissParticleRef > L1TkEtMissParticleVectorRef ;
}
#endif
