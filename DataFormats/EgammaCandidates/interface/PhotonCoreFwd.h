#ifndef EgammaReco_PhotonCoreFwd_h
#define EgammaReco_PhotonCoreFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class PhotonCore;

  /// collectin of PhotonCore objects
  typedef std::vector<PhotonCore> PhotonCoreCollection;

  /// reference to an object in a collection of PhotonCore objects
  typedef edm::Ref<PhotonCoreCollection> PhotonCoreRef;

  /// reference to a collection of PhotonCore objects
  typedef edm::RefProd<PhotonCoreCollection> PhotonCoreRefProd;

  /// vector of objects in the same collection of PhotonCore objects
  typedef edm::RefVector<PhotonCoreCollection> PhotonCoreRefVector;

  /// iterator over a vector of reference to PhotonCore objects
  typedef PhotonCoreRefVector::iterator photonCore_iterator;
}

#endif
