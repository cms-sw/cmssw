#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/RecoAlgos/src/ObjectCountFilters.h"
#include "PhysicsTools/RecoAlgos/src/ObjectPairFilters.h"
#include "PhysicsTools/RecoAlgos/src/ObjectSelectors.h"
#include "PhysicsTools/RecoAlgos/src/ObjectUtilities.h"

DEFINE_SEAL_MODULE();

 // filters
DEFINE_ANOTHER_FWK_MODULE( TrackCountFilter )
DEFINE_ANOTHER_FWK_MODULE( ElectronCountFilter )
DEFINE_ANOTHER_FWK_MODULE( PhotonCountFilter )
DEFINE_ANOTHER_FWK_MODULE( MuonCountFilter )
DEFINE_ANOTHER_FWK_MODULE( CaloJetCountFilter )
DEFINE_ANOTHER_FWK_MODULE( PtMinTrackCountFilter )
DEFINE_ANOTHER_FWK_MODULE( PtMinElectronCountFilter )
DEFINE_ANOTHER_FWK_MODULE( PtMinPhotonCountFilter )
DEFINE_ANOTHER_FWK_MODULE( PtMinMuonCountFilter )
DEFINE_ANOTHER_FWK_MODULE( EtMinCaloJetCountFilter )
DEFINE_ANOTHER_FWK_MODULE( EtaEtMinCaloJetCountFilter )
DEFINE_ANOTHER_FWK_MODULE( ElectronPairMassFilter )
DEFINE_ANOTHER_FWK_MODULE( MuonPairMassFilter )

    // selectors
DEFINE_ANOTHER_FWK_MODULE( AnyTrackSelector )
DEFINE_ANOTHER_FWK_MODULE( PtMinTrackSelector )
DEFINE_ANOTHER_FWK_MODULE( PtMinElectronSelector )
DEFINE_ANOTHER_FWK_MODULE( EtMinPhotonSelector )
DEFINE_ANOTHER_FWK_MODULE( EtMinCaloJetSelector )
DEFINE_ANOTHER_FWK_MODULE( EtMinCaloJetShallowCloneSelector )
DEFINE_ANOTHER_FWK_MODULE( EtaPtMinElectronSelector )
DEFINE_ANOTHER_FWK_MODULE( LargestPtTrackSelector )
DEFINE_ANOTHER_FWK_MODULE( LargestEtCaloJetSelector )
DEFINE_ANOTHER_FWK_MODULE( LargestEtCaloJetShallowCloneSelector )
DEFINE_ANOTHER_FWK_MODULE( MassWindowTrackSelector )
DEFINE_ANOTHER_FWK_MODULE( ConfigTrackSelector )

 // other utilities
DEFINE_ANOTHER_FWK_MODULE( TrackMerger )
DEFINE_ANOTHER_FWK_MODULE( MuonMerger )
DEFINE_ANOTHER_FWK_MODULE( ElectronMerger )
DEFINE_ANOTHER_FWK_MODULE( PhotonMerger )
DEFINE_ANOTHER_FWK_MODULE( CaloJetMerger )
DEFINE_ANOTHER_FWK_MODULE( TrackRecoverer )
