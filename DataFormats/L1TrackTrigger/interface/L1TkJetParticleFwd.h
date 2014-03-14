#ifndef L1TkTrigger_L1JetParticleFwd_h
#define L1TkTrigger_L1JetParticleFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkJetParticleFwd
// 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1TkJetParticle ;

   typedef edm::RefProd< L1TkJetParticle > L1TkJetParticleRefProd ;

   typedef std::vector< L1TkJetParticle > L1TkJetParticleCollection ;

   typedef edm::Ref< L1TkJetParticleCollection > L1TkJetParticleRef ;
   typedef edm::RefVector< L1TkJetParticleCollection > L1TkJetParticleRefVector ;
   typedef std::vector< L1TkJetParticleRef > L1TkJetParticleVectorRef ;
}

#endif


