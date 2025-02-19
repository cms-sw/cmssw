#ifndef L1Trigger_L1ParticleMapFwd_h
#define L1Trigger_L1ParticleMapFwd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1ParticleMapFwd
// 
/**\class L1ParticleMapCollection \file L1ParticleMapFwd.h DataFormats/L1Trigger/interface/L1ParticleMapFwd.h \author Werner Sun

 Description: typedefs for L1ParticleMapCollection and associated containers.
*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 14:28:43 EDT 2006
// $Id: L1ParticleMapFwd.h,v 1.2 2007/04/02 08:03:13 wsun Exp $
//

// system include files

// user include files

// forward declarations
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1extra {

   class L1ParticleMap ;

   typedef std::vector< L1ParticleMap > L1ParticleMapCollection ;

   typedef edm::Ref< L1ParticleMapCollection > L1ParticleMapRef ;
   typedef edm::RefVector< L1ParticleMapCollection > L1ParticleMapRefVector ;
}

#endif
