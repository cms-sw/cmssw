#ifndef L1TkTrigger_L1TauParticleFwd_h
#define L1TkTrigger_L1TaunParticleFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkTauParticleFwd
// 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1TkTauParticle ;

   typedef std::vector< L1TkTauParticle > L1TkTauParticleCollection ;

   typedef edm::Ref< L1TkTauParticleCollection > L1TkTauParticleRef ;
   typedef edm::RefVector< L1TkTauParticleCollection > L1TkTauParticleRefVector ;
   typedef std::vector< L1TkTauParticleRef > L1TkTauParticleVectorRef ;
}

#endif


