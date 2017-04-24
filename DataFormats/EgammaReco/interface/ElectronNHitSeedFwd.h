#ifndef EGammaReco_ElectronNHitSeedFwd_h
#define EGammaReco_ElectronNHitSeedFwd_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ElectronNHitSeed;
  /// collection of ElectronNHitSeed objects
  typedef std::vector<ElectronNHitSeed> ElectronNHitSeedCollection;
  /// reference to an object in a collection of ElectronNHitSeed objects
  typedef edm::Ref<ElectronNHitSeedCollection> ElectronNHitSeedRef;
  /// reference to a collection of ElectronNHitSeed objects
  typedef edm::RefProd<ElectronNHitSeedCollection> ElectronNHitSeedRefProd;
  /// vector of objects in the same collection of ElectronNHitSeed objects
  typedef edm::RefVector<ElectronNHitSeedCollection> ElectronNHitSeedRefVector;
  /// iterator over a vector of reference to ElectronNHitSeed objects
  typedef ElectronNHitSeedRefVector::iterator electronnithitseed_iterator;
}

#endif
