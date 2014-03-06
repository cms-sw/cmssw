#ifndef L1TkTrigger_L1ElectronParticleFwd_h
#define L1TkTrigger_L1ElectronParticleFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkElectronParticleFwd
// 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1TkElectronParticle ;

   typedef std::vector< L1TkElectronParticle > L1TkElectronParticleCollection ;

   typedef edm::Ref< L1TkElectronParticleCollection > L1TkElectronParticleRef ;
   typedef edm::RefVector< L1TkElectronParticleCollection > L1TkElectronParticleRefVector ;
   typedef std::vector< L1TkElectronParticleRef > L1TkElectronParticleVectorRef ;
}

#endif


