#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/SingleElementCollectionSelector.h"
#include "PhysicsTools/RecoAlgos/interface/AnySelector.h"
#include "PhysicsTools/RecoAlgos/interface/PtMinSelector.h"
#include "PhysicsTools/RecoAlgos/interface/SortCollectionSelector.h"
#include "PhysicsTools/RecoAlgos/interface/WindowCollectionSelector.h"
#include "PhysicsTools/RecoAlgos/interface/PtComparator.h"
#include "PhysicsTools/RecoAlgos/interface/MasslessInvariantMass.h"
#include "PhysicsTools/RecoAlgos/src/RecoModules.h"

typedef ObjectSelector<SingleElementCollectionSelector<reco::TrackCollection, 
						       AnySelector<reco::Track> > 
                      > AnyTrackSelector;
typedef ObjectSelector<SingleElementCollectionSelector<reco::TrackCollection,
						       PtMinSelector<reco::Track> >
                      > PtMinTrackSelector;
typedef ObjectSelector<SortCollectionSelector<reco::TrackCollection, 
					      PtInverseComparator<reco::Track> > 
                     > LargestPtTrackSelector;
typedef ObjectSelector<WindowCollectionSelector<reco::TrackCollection, 
						MasslessInvariantMass<reco::Track> >
                      > MassWindowTrackSelector;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( AnyTrackSelector );
DEFINE_ANOTHER_FWK_MODULE( PtMinTrackSelector );
DEFINE_ANOTHER_FWK_MODULE( LargestPtTrackSelector )
DEFINE_ANOTHER_FWK_MODULE( MassWindowTrackSelector )
namespace reco {
  namespace modules {
DEFINE_ANOTHER_FWK_MODULE( TrackMerger );
DEFINE_ANOTHER_FWK_MODULE( MuonMerger );
DEFINE_ANOTHER_FWK_MODULE( ElectronMerger );
DEFINE_ANOTHER_FWK_MODULE( PhotonMerger );
DEFINE_ANOTHER_FWK_MODULE( TrackRecoverer );
  }
}
