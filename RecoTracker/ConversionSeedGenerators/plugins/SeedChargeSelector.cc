#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "SeedChargeSelector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<TrajectorySeedCollection, ::SeedChargeSelector> SeedChargeSelector;

    DEFINE_FWK_MODULE(SeedChargeSelector);
  }  // namespace modules
}  // namespace reco
