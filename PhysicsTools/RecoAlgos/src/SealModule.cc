#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/RecoAlgos/interface/SingleTrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/AnySelector.h"
#include "PhysicsTools/RecoAlgos/src/LargestPtTrackSelector.h"
#include "PhysicsTools/RecoAlgos/src/RecoModules.h"

typedef SingleTrackSelector<AnySelector<reco::Track> > AnyTrackSelector;

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
