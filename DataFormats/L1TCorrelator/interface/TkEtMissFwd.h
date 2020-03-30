#ifndef TkTrigger_TkEtMissFwd_h
#define TkTrigger_TkEtMissFwd_h
// Package:     L1Trigger
// Class  :     TkEtMissFwd

// system include files
// user include files
// forward declarations
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {
  class TkEtMiss;
  typedef std::vector<TkEtMiss> TkEtMissCollection;
  //typedef edm::RefProd< TkEtMiss > TkEtMissRefProd ;
  //typedef edm::Ref< TkEtMissCollection > TkEtMissRef ;
  //typedef edm::RefVector< TkEtMissCollection > TkEtMissRefVector ;
  //typedef std::vector< TkEtMissRef > TkEtMissVectorRef ;
}  // namespace l1t
#endif
