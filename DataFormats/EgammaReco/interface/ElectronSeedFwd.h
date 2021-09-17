#ifndef EGammaReco_ElectronSeedFwd_h
#define EGammaReco_ElectronSeedFwd_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ElectronSeed;
  /// collection of ElectronSeed objects
  typedef std::vector<ElectronSeed> ElectronSeedCollection;
  /// reference to an object in a collection of ElectronSeed objects
  typedef edm::Ref<ElectronSeedCollection> ElectronSeedRef;
  /// reference to a collection of ElectronSeed objects
  typedef edm::RefProd<ElectronSeedCollection> ElectronSeedRefProd;
  /// vector of objects in the same collection of ElectronSeed objects
  typedef edm::RefVector<ElectronSeedCollection> ElectronSeedRefVector;
}  // namespace reco

#endif
