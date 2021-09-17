#ifndef EgammaReco_PhotonFwd_h
#define EgammaReco_PhotonFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Photon;

  /// collectin of Photon objects
  typedef std::vector<Photon> PhotonCollection;

  /// reference to an object in a collection of Photon objects
  typedef edm::Ref<PhotonCollection> PhotonRef;

  /// reference to a collection of Photon objects
  typedef edm::RefProd<PhotonCollection> PhotonRefProd;

  /// vector of objects in the same collection of Photon objects
  typedef edm::RefVector<PhotonCollection> PhotonRefVector;

  /// iterator over a vector of reference to Photon objects
  typedef PhotonRefVector::iterator photon_iterator;
}  // namespace reco

#endif
