#ifndef L1Trigger_L1EtMissParticleFwd_h
#define L1Trigger_L1EtMissParticleFwd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtMissParticleFwd
// 
/**\class L1EtMissParticleRef \file L1EtMissParticleFwd.h DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h \author Werner Sun

 Description: typedefs for L1EtMissParticleRef and associated containers.
*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 14:28:43 EDT 2006
// $Id: L1EtMissParticleFwd.h,v 1.3 2007/12/18 03:26:49 wsun Exp $
//

// system include files

// user include files

// forward declarations
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1EtMissParticle ;

   typedef edm::RefProd< L1EtMissParticle > L1EtMissParticleRefProd ;

   typedef std::vector< L1EtMissParticle > L1EtMissParticleCollection ;

   typedef edm::Ref< L1EtMissParticleCollection > L1EtMissParticleRef ;
   typedef edm::RefVector< L1EtMissParticleCollection > L1EtMissParticleRefVector ;
   typedef std::vector< L1EtMissParticleRef > L1EtMissParticleVectorRef ;
}

#endif
