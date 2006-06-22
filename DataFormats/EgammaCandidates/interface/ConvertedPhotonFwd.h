#ifndef EgammaReco_ConvertedPhotonFwd_h
#define EgammaReco_ConvertedPhotonFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ConvertedPhoton;

  /// collectin of ConvertedPhoton objects
  typedef std::vector<ConvertedPhoton> ConvertedPhotonCollection;

  /// reference to an object in a collection of ConvertedPhoton objects
  typedef edm::Ref<ConvertedPhotonCollection> ConvertedPhotonRef;

  /// reference to a collection of ConvertedPhoton objects
  typedef edm::RefProd<ConvertedPhotonCollection> ConvertedPhotonRefProd;

  /// vector of objects in the same collection of ConvertedPhoton objects
  typedef edm::RefVector<ConvertedPhotonCollection> ConvertedPhotonRefVector;

  /// iterator over a vector of reference to ConvertedPhoton objects
  typedef ConvertedPhotonRefVector::iterator convPhoton_iterator;
}

#endif
