#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelectorPlusEvent.h"
#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "PhysicsTools/RecoAlgos/plugins/TrackWithVertexSelector.h"

typedef ObjectSelector<
  SingleElementCollectionSelectorPlusEvent<
          reco::TrackCollection,
          reco::modules::TrackWithVertexSelector,
          reco::TrackRefVector 
          >
  > TrackWithVertexRefSelectorProducer;

DEFINE_FWK_MODULE(TrackWithVertexRefSelectorProducer);
