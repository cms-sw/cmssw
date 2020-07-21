#ifndef DataFormats_EgammaReco_EgTrigSumObjFwd_h
#define DataFormats_EgammaReco_EgTrigSumObjFwd_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class EgTrigSumObj;

  /// collection of SuperCluser objectr
  typedef std::vector<EgTrigSumObj> EgTrigSumObjCollection;
  /// reference to an object in a collection of EgTrigSumObj objects
  typedef edm::Ref<EgTrigSumObjCollection> EgTrigSumObjRef;
  /// reference to a collection of EgTrigSumObj objects
  typedef edm::RefProd<EgTrigSumObjCollection> EgTrigSumObjRefProd;
  /// vector of references to objects in the same colletion of EgTrigSumObj objects
  typedef edm::RefVector<EgTrigSumObjCollection> EgTrigSumObjRefVector;
  /// iterator over a vector of reference to EgTrigSumObjs
  typedef EgTrigSumObjRefVector::iterator egTrigSumObj_iterator;
}

#endif
