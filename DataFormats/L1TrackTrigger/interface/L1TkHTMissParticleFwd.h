#ifndef L1TkTrigger_L1TkHTMissParticleFwd_h
#define L1TkTrigger_L1TkHTMissParticleFwd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkHTMissParticleFwd
// 

// system include files

// user include files

// forward declarations
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1TkHTMissParticle ;

   //typedef edm::RefProd< L1TkHTMissParticle > L1TkHTMissParticleRefProd ;

   typedef std::vector< L1TkHTMissParticle > L1TkHTMissParticleCollection ;

   //typedef edm::Ref< L1TkHTMissParticleCollection > L1TkHTMissParticleRef ;
   //typedef edm::RefVector< L1TkHTMissParticleCollection > L1TkHTMissParticleRefVector ;
   //typedef std::vector< L1TkHTMissParticleRef > L1TkHTMissParticleVectorRef ;
}

#endif
