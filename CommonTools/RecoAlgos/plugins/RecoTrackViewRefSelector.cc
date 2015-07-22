#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorStreamProducer.h"
#include "CommonTools/RecoAlgos/interface/RecoTrackViewRefSelector.h"

namespace reco {
  typedef ObjectSelectorStreamProducer<RecoTrackViewRefSelector, edm::RefToBaseVector<reco::Track>> RecoTrackViewRefSelector;
  DEFINE_FWK_MODULE(RecoTrackViewRefSelector);
}
