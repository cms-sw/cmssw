#ifndef L1TkTrigger_L1EmParticleFwd_h
#define L1TkTrigger_L1EmParticleFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEmParticleFwd
// 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1TkEmParticle ;

   typedef std::vector< L1TkEmParticle > L1TkEmParticleCollection ;

   typedef edm::Ref< L1TkEmParticleCollection > L1TkEmParticleRef ;
   typedef edm::RefVector< L1TkEmParticleCollection > L1TkEmParticleRefVector ;
   typedef std::vector< L1TkEmParticleRef > L1TkEmParticleVectorRef ;
}

#endif


