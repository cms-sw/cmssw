#ifndef L1Trigger_L1HFRingsFwd_h
#define L1Trigger_L1HFRingsFwd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1HFRingsFwd
//
/**\class L1HFRingsCollection \file L1HFRingsFwd.h DataFormats/L1Trigger/interface/L1HFRingsFwd.h \author Werner Sun

 Description: typedefs for L1HFRingsCollection and associated containers.
*/
//
// Original Author:  Werner Sun
//         Created:  Fri Mar 20 16:16:25 CET 2009
// $$
//

// system include files

// user include files

// forward declarations
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1extra {

  class L1HFRings;

  typedef std::vector<L1HFRings> L1HFRingsCollection;

  typedef edm::Ref<L1HFRingsCollection> L1HFRingsRef;
  typedef edm::RefVector<L1HFRingsCollection> L1HFRingsRefVector;
  typedef std::vector<L1HFRingsRef> L1HFRingsVectorRef;
}  // namespace l1extra

#endif
