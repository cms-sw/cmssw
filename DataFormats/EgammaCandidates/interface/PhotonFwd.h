#ifndef EgammaReco_PhotonFwd_h
#define EgammaReco_PhotonFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Photon;

  /// collection of Photon objects
  typedef std::vector<Photon> PhotonCollection;

  /// reference to an object in a collection of Photon objects
  typedef edm::Ref<PhotonCollection> PhotonRef;

  /// reference to a collection of Photon objects
  typedef edm::RefProd<PhotonCollection> PhotonRefProd;

  /// vector of references to objects in the same collection of Photon objects
  typedef edm::RefVector<PhotonCollection> PhotonRefVector;

  /// iterator over a vector of references to Photon objects
  typedef PhotonRefVector::iterator photon_iterator;
}

#endif
