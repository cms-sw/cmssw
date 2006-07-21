#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/SingleTrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/AnySelector.h"
#include "PhysicsTools/RecoAlgos/interface/SortCollectionSelector.h"
#include "PhysicsTools/RecoAlgos/interface/PtComparator.h"
#include "PhysicsTools/RecoAlgos/src/RecoModules.h"

typedef SingleTrackSelector<AnySelector<reco::Track> > AnyTrackSelector;
typedef TrackSelector<SortCollectionSelector<reco::TrackCollection, 
					     PtInverseComparator<reco::Track> > 
                     > LargestPtTrackSelector;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( AnyTrackSelector );
DEFINE_ANOTHER_FWK_MODULE( LargestPtTrackSelector )
namespace reco {
  namespace modules {
DEFINE_ANOTHER_FWK_MODULE( TrackMerger );
DEFINE_ANOTHER_FWK_MODULE( MuonMerger );
DEFINE_ANOTHER_FWK_MODULE( ElectronMerger );
DEFINE_ANOTHER_FWK_MODULE( PhotonMerger );
DEFINE_ANOTHER_FWK_MODULE( TrackRecoverer );
  }
}
