#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"
#include "PhysicsTools/UtilAlgos/interface/CollectionRecoverer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "PhysicsTools/Utilities/interface/MasslessInvariantMass.h"
#include "PhysicsTools/Utilities/interface/AnySelector.h"
#include "PhysicsTools/Utilities/interface/PtMinSelector.h"
#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "PhysicsTools/Utilities/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SortCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/WindowCollectionSelector.h"
#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"

namespace reco {
  namespace modules {
    /// Framework module to merge an arbitray number of reco::TrackCollection
    typedef Merger<reco::TrackCollection>    TrackMerger;
    /// Framework module to merge an arbitray number of reco::MuonCollection
    typedef Merger<reco::MuonCollection>     MuonMerger;
    /// Framework module to merge an arbitray number of reco::ElectronCollection
    typedef Merger<reco::ElectronCollection> ElectronMerger;
    /// Framework module to merge an arbitray number of reco::PhotonCollection
    typedef Merger<reco::PhotonCollection>   PhotonMerger;
    /// Recover a collection of reco::Track
    typedef CollectionRecoverer<reco::TrackCollection> TrackRecoverer; 

    /// select all tracks
    typedef ObjectSelector<
              SingleElementCollectionSelector<
                reco::TrackCollection, 
                AnySelector<reco::Track> 
              > 
            > AnyTrackSelector;
    /// select tracks above a give pt
    typedef ObjectSelector<
              SingleElementCollectionSelector<
                reco::TrackCollection,
                PtMinSelector<reco::Track> 
              >
                          
            > PtMinTrackSelector;
    /// select the N tracks with highest pt
    typedef ObjectSelector<
              SortCollectionSelector<
                reco::TrackCollection, 
		PtInverseComparator<reco::Track> 
              > 
            > LargestPtTrackSelector;
    /// select track pairs within a given mass window
    typedef ObjectSelector<
              WindowCollectionSelector<
                reco::TrackCollection, 
		MasslessInvariantMass<reco::Track> 
              >
            > MassWindowTrackSelector;
    /// configurable single track selector
    typedef ObjectSelector<
              SingleElementCollectionSelector<
                reco::TrackCollection, 
                SingleObjectSelector<reco::Track> 
              >
            > ConfigTrackSelector;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( AnyTrackSelector );
DEFINE_ANOTHER_FWK_MODULE( PtMinTrackSelector );
DEFINE_ANOTHER_FWK_MODULE( LargestPtTrackSelector )
DEFINE_ANOTHER_FWK_MODULE( MassWindowTrackSelector )
DEFINE_ANOTHER_FWK_MODULE( TrackMerger );
DEFINE_ANOTHER_FWK_MODULE( MuonMerger );
DEFINE_ANOTHER_FWK_MODULE( ElectronMerger );
DEFINE_ANOTHER_FWK_MODULE( PhotonMerger );
DEFINE_ANOTHER_FWK_MODULE( TrackRecoverer );
DEFINE_ANOTHER_FWK_MODULE( ConfigTrackSelector );
  }
}
