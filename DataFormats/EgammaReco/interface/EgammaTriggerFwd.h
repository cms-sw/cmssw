#ifndef EgammaReco_EgammaTriggerFwd_h
#define EgammaReco_EgammaTriggerFwd_h
// $Id: EgammaTriggerFwd.h,v 1.1 2006/04/09 15:40:40 rahatlou Exp $
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class EgammaTrigger;

  /// collection of EgammaTrigger objects
  typedef std::vector<EgammaTrigger> EgammaTriggerCollection;

  /// persistent reference to EgammaTrigger objects
  typedef edm::Ref<EgammaTriggerCollection> EgammaTriggerRef;

  /// reference to a EgammaTrigger collection
  typedef edm::RefProd<EgammaTriggerCollection> EgammaTriggerRefProd;

  /// vector of references to EgammaTrigger objects in the same collection
  typedef edm::RefVector<EgammaTriggerCollection> EgammaTriggerRefVector;

  /// iterator over a vector of references to EgammaTrigger objects
  typedef EgammaTriggerRefVector::iterator egammaTrigger_iterator;
}

#endif
