#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelectorPlusEvent.h"
#include "PhysicsTools/RecoAlgos/interface/TrackWithVertexSelector.h"

namespace reco { 
  namespace modules {

typedef ObjectSelector<
  SingleElementCollectionSelectorPlusEvent<
          reco::TrackCollection,
          ::TrackWithVertexSelector,
          reco::TrackRefVector 
          >
  > TrackWithVertexRefSelector;

DEFINE_FWK_MODULE(TrackWithVertexRefSelector);

} }
