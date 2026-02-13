#ifndef DataFormatsL1TCorrelator_TkHTMissFwd_h
#define DataFormatsL1TCorrelator_TkHTMissFwd_h

// forward declarations
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {
  namespace io_v1 {
    class TkHTMiss;
  }
  using TkHTMiss = io_v1::TkHTMiss;
  typedef std::vector<TkHTMiss> TkHTMissCollection;
}  // namespace l1t

#endif
