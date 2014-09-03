#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"

#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelectorPlusEvent.h"
#include "CommonTools/RecoAlgos/interface/TrackWithVertexSelector.h"

namespace reco { 
  namespace modules {

typedef ObjectSelectorStream<
  SingleElementCollectionSelectorPlusEvent<
          reco::TrackCollection,
          ::TrackWithVertexSelector,
          reco::TrackRefVector 
          >,
  reco::TrackRefVector > TrackWithVertexRefSelector;

DEFINE_FWK_MODULE(TrackWithVertexRefSelector);

} }
