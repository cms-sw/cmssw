#ifndef L1TkTrigger_L1GlbMuonParticleFwd_h
#define L1TkTrigger_L1GlbMuonParticleFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkGlbMuonParticleFwd
// 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1t {

   class L1TkGlbMuonParticle ;

   typedef std::vector< L1TkGlbMuonParticle > L1TkGlbMuonParticleCollection ;

   typedef edm::Ref< L1TkGlbMuonParticleCollection > L1TkGlbMuonParticleRef ;
   typedef edm::RefVector< L1TkGlbMuonParticleCollection > L1TkGlbMuonParticleRefVector ;
   typedef std::vector< L1TkGlbMuonParticleRef > L1TkGlbMuonParticleVectorRef ;
}

#endif


