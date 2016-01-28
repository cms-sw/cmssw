#ifndef __L1TMUON_DTBUNCHCROSSINGCLEANER_H__
#define __L1TMUON_DTBUNCHCROSSINGCLEANER_H__
//
// Class: L1TMuon::DTBunchCrossingCleaner
//
// Info: This class analyzes the output of a DT chamber and produces
//       a reduced set of trigger primitives combining theta and phi
//       trigger primitives that are likely to be associated.
//
// Author: L. Gray (FNAL)
//
#include <vector>
#include "L1Trigger/L1TMuon/interface/deprecate/MuonTriggerPrimitive.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm{
  class ParameterSet;
}

using namespace L1TMuon;

namespace L1TwinMux {

  class DTBunchCrossingCleaner {
  public:
    DTBunchCrossingCleaner();
    ~DTBunchCrossingCleaner() {}

    TriggerPrimitiveCollection clean(const TriggerPrimitiveCollection&)
      const;

  private:
    const int bx_window_size = 1;
  };

}

#endif
