#ifndef L1Trigger_L1MuonParticleExtendedFwd_h
#define L1Trigger_L1MuonParticleExtendedFwd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1MuonParticleExtendedFwd
// 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1MuonParticleExtended ;

   typedef std::vector< L1MuonParticleExtended > L1MuonParticleExtendedCollection ;

   typedef edm::Ref< L1MuonParticleExtendedCollection > L1MuonParticleExtendedRef ;
   typedef edm::RefVector< L1MuonParticleExtendedCollection > L1MuonParticleExtendedRefVector ;
   typedef std::vector< L1MuonParticleExtendedRef > L1MuonParticleExtendedVectorRef ;
}

#endif
