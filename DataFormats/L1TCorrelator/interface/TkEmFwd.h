#ifndef DataFormatsL1TCorrelator_TkEmFwd_h
#define DataFormatsL1TCorrelator_TkEmFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEmFwd
//

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/L1Trigger/interface/RegionalOutput.h"

namespace l1t {

  namespace io_v1 {
    class TkEm;
  }
  using TkEm = io_v1::TkEm;

  typedef std::vector<TkEm> TkEmCollection;

  typedef edm::Ref<TkEmCollection> TkEmRef;
  typedef edm::RefVector<TkEmCollection> TkEmRefVector;
  typedef std::vector<TkEmRef> TkEmVectorRef;
  typedef l1t::RegionalOutput<l1t::TkEmCollection> TkEmRegionalOutput;

}  // namespace l1t

#endif
