#ifndef L1Trigger_L1EmParticleFwd_h
#define L1Trigger_L1EmParticleFwd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EmParticleFwd
// 
/**\class L1EmParticleFwd L1EmParticleFwd.h DataFormats/L1Trigger/interface/L1EmParticleFwd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 14:28:43 EDT 2006
// $Id: L1EmParticleFwd.h,v 1.2 2006/07/26 00:05:39 wsun Exp $
//

// system include files

// user include files

// forward declarations
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1EmParticle ;

   typedef std::vector< L1EmParticle > L1EmParticleCollection ;

   typedef edm::Ref< L1EmParticleCollection > L1EmParticleRef ;
   typedef edm::RefVector< L1EmParticleCollection > L1EmParticleRefVector ;
   typedef std::vector< L1EmParticleRef > L1EmParticleVectorRef ;
}

#endif
