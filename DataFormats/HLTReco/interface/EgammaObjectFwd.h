#ifndef DataFormats_HLTReco_EgammaObjectFwd_h
#define DataFormats_HLTReco_EgammaObjectFwd_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace trigger {
  class EgammaObject;

  typedef std::vector<EgammaObject> EgammaObjectCollection;
  typedef edm::Ref<EgammaObjectCollection> EgammaObjectRef;
  typedef edm::RefProd<EgammaObjectCollection> EgammaObjectRefProd;
  typedef edm::RefVector<EgammaObjectCollection> EgammaObjectRefVector;
  typedef EgammaObjectRefVector::iterator EgammaObjectIterator;
}  // namespace trigger

#endif
