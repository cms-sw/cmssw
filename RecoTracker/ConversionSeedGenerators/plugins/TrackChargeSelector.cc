#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "TrackChargeSelector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<TrackCollection, ::TrackChargeSelector> TrackChargeSelector;

    DEFINE_FWK_MODULE(TrackChargeSelector);
  }  // namespace modules
}  // namespace reco
