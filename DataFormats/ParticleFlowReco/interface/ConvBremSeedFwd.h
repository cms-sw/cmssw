#ifndef EGammaReco_ConvBremSeedFwd_h
#define EGammaReco_ConvBremSeedFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ConvBremSeed;
  /// collectin of ConvBremSeed objects
  typedef std::vector<ConvBremSeed> ConvBremSeedCollection;
  /// reference to an object in a collection of ConvBremSeed objects
  typedef edm::Ref<ConvBremSeedCollection> ConvBremSeedRef;
  /// reference to a collection of ConvBremSeed objects
  typedef edm::RefProd<ConvBremSeedCollection> ConvBremSeedRefProd;
  /// vector of objects in the same collection of ConvBremSeed objects
  typedef edm::RefVector<ConvBremSeedCollection> ConvBremSeedRefVector;
  /// iterator over a vector of reference to ConvBremSeed objects
  typedef ConvBremSeedRefVector::iterator convbremphltseed_iterator;
}

#endif
