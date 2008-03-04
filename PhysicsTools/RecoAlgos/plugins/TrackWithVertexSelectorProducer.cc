#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelectorPlusEvent.h"
#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "PhysicsTools/RecoAlgos/plugins/TrackWithVertexSelector.h"

namespace reco { 
  namespace modules {
typedef ObjectSelector<
  SingleElementCollectionSelectorPlusEvent<
          reco::TrackCollection,
          reco::modules::TrackWithVertexSelector
          >
  > TrackWithVertexSelectorProducer;
  DEFINE_FWK_MODULE(TrackWithVertexSelectorProducer);
} }
