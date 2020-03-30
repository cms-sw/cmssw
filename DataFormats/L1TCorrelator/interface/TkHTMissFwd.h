#ifndef TkTrigger_TkHTMissFwd_h
#define TkTrigger_TkHTMissFwd_h
// Package:     L1Trigger
// Class  :     TkHTMissFwd

// system include files
// user include files

// forward declarations
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {
  class TkHTMiss;
  typedef std::vector<TkHTMiss> TkHTMissCollection;
  //typedef edm::RefProd< TkHTMiss > TkHTMissRefProd ;
  //typedef edm::Ref< TkHTMissCollection > TkHTMissRef ;
  //typedef edm::RefVector< TkHTMissCollection > TkHTMissRefVector ;
  //typedef std::vector< TkHTMissRef > TkHTMissVectorRef ;
}  // namespace l1t

#endif
