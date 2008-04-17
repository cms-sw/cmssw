#ifndef PhotonIDFwd_h
#define PhotonIDFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"

namespace reco {
  class PhotonID;

  /// collection of PhotonID objects
  typedef std::vector<PhotonID> PhotonIDCollection;

  /// reference to an object in a collection of PhotonID objects
  typedef edm::Ref<PhotonIDCollection> PhotonIDRef;

  /// reference to a collection of PhotonID objects
  typedef edm::RefProd<PhotonIDCollection> PhotonIDRefProd;

  /// vector of objects in the same collection of PhotonID objects
  typedef edm::RefVector<PhotonIDCollection> PhotonIDRefVector;

  /// iterator over a vector of reference to PhotonID objects
  typedef PhotonIDRefVector::iterator photonID_iterator;
}

#endif
