#ifndef EGammaReco_ElectronPixelSeedFwd_h
#define EGammaReco_ElectronPixelSeedFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ElectronPixelSeed;
  /// collectin of ElectronPixelSeed objects
  typedef std::vector<ElectronPixelSeed> ElectronPixelSeedCollection;
  /// reference to an object in a collection of ElectronPixelSeed objects
  typedef edm::Ref<ElectronPixelSeedCollection> ElectronPixelSeedRef;
  /// reference to a collection of ElectronPixelSeed objects
  typedef edm::RefProd<ElectronPixelSeedCollection> ElectronPixelSeedRefProd;
  /// vector of objects in the same collection of ElectronPixelSeed objects
  typedef edm::RefVector<ElectronPixelSeedCollection> ElectronPixelSeedRefVector;
  /// iterator over a vector of reference to ElectronPixelSeed objects
  typedef ElectronPixelSeedRefVector::iterator electronephltseed_iterator;
}

#endif
