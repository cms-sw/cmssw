#ifndef L1TkTrigger_L1MuonParticleFwd_h
#define L1TkTrigger_L1MuonParticleFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkMuonParticleFwd
// 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1TkMuonParticle ;

   typedef std::vector< L1TkMuonParticle > L1TkMuonParticleCollection ;

   typedef edm::Ref< L1TkMuonParticleCollection > L1TkMuonParticleRef ;
   typedef edm::RefVector< L1TkMuonParticleCollection > L1TkMuonParticleRefVector ;
   typedef std::vector< L1TkMuonParticleRef > L1TkMuonParticleVectorRef ;
}

#endif


