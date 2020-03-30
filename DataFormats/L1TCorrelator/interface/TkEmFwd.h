#ifndef TkTrigger_L1EmFwd_h
#define TkTrigger_L1EmFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEmFwd
//

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {

  class TkEm;

  typedef std::vector<TkEm> TkEmCollection;

  typedef edm::Ref<TkEmCollection> TkEmRef;
  typedef edm::RefVector<TkEmCollection> TkEmRefVector;
  typedef std::vector<TkEmRef> TkEmVectorRef;
}  // namespace l1t

#endif
