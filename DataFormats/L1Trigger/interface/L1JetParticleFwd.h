#ifndef L1Trigger_L1JetParticleFwd_h
#define L1Trigger_L1JetParticleFwd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1JetParticleFwd
// 
/**\class L1JetParticleCollection \file L1JetParticleFwd.h DataFormats/L1Trigger/interface/L1JetParticleFwd.h \author Werner Sun

 Description: typedefs for L1JetParticleCollection and associated containers.
*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 14:28:43 EDT 2006
// $Id: L1JetParticleFwd.h,v 1.4 2007/04/02 08:03:13 wsun Exp $
//

// system include files

// user include files

// forward declarations
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1JetParticle ;

   typedef std::vector< L1JetParticle > L1JetParticleCollection ;

   typedef edm::Ref< L1JetParticleCollection > L1JetParticleRef ;
   typedef edm::RefVector< L1JetParticleCollection > L1JetParticleRefVector ;
   typedef std::vector< L1JetParticleRef > L1JetParticleVectorRef ;
}

#endif
