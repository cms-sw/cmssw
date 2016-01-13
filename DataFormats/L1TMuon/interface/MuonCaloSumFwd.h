#ifndef __GMTInputCaloSumFwd_h
#define __GMTInputCaloSumFwd_h

#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
  class MuonCaloSum;
  typedef BXVector<MuonCaloSum> MuonCaloSumBxCollection;
}

#endif
