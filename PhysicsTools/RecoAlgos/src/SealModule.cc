#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"
#include "PhysicsTools/UtilAlgos/interface/CollectionRecoverer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "PhysicsTools/Utilities/interface/MasslessInvariantMass.h"
#include "PhysicsTools/Utilities/interface/AnySelector.h"
#include "PhysicsTools/Utilities/interface/PtMinSelector.h"
#include "PhysicsTools/Utilities/interface/EtMinSelector.h"
#include "PhysicsTools/Utilities/interface/EtaRangeSelector.h"
#include "PhysicsTools/Utilities/interface/AndSelector.h"
#include "PhysicsTools/Utilities/interface/OrSelector.h"
#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SortCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/WindowCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/Parser/interface/SingleObjectSelector.h"
#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"

namespace reco {
  namespace modules {
    /// filter on track number
    typedef ObjectCountFilter<reco::TrackCollection> TrackCountFilter;

    /// filter on electron number
    typedef ObjectCountFilter<reco::ElectronCollection> ElectronCountFilter;

    /// filter on photon number
    typedef ObjectCountFilter<reco::PhotonCollection> PhotonCountFilter;

    /// filter on muon number
    typedef ObjectCountFilter<reco::MuonCollection> MuonCountFilter;

    /// filter on calo jet number
    typedef ObjectCountFilter<reco::CaloJetCollection> CaloJetCountFilter;

    /// filter on basic jet number
    typedef ObjectCountFilter<reco::BasicJetCollection> BasicJetCountFilter;

    /// filter on generator jet number
    typedef ObjectCountFilter<reco::GenJetCollection> GenJetCountFilter;

    /// filter on track number
    typedef ObjectCountFilter<reco::TrackCollection, PtMinSelector<reco::Track> > PtMinTrackCountFilter;

    /// filter on electron number
    typedef ObjectCountFilter<reco::ElectronCollection, PtMinSelector<reco::Electron> > PtMinElectronCountFilter;

    /// filter on photon number
    typedef ObjectCountFilter<reco::PhotonCollection, PtMinSelector<reco::Photon> > PtMinPhotonCountFilter;

    /// filter on muon number
    typedef ObjectCountFilter<reco::MuonCollection, PtMinSelector<reco::Muon> > PtMinMuonCountFilter;

    /// filter on calo jets
    typedef ObjectCountFilter<reco::CaloJetCollection, EtMinSelector<reco::CaloJet> > EtMinCaloJetCountFilter;

    /// filter on basic jets
    typedef ObjectCountFilter<reco::BasicJetCollection, EtMinSelector<reco::BasicJet> > EtMinBasicJetCountFilter;

    /// filter on generator jets
    typedef ObjectCountFilter<reco::GenJetCollection, EtMinSelector<reco::GenJet> > EtMinGenJetCountFilter;

    /// filter on calo jets
    typedef ObjectCountFilter<reco::CaloJetCollection, 
			      AndSelector<EtMinSelector<reco::CaloJet>,
					  EtaRangeSelector<reco::CaloJet> > 
                             > EtaEtMinCaloJetCountFilter;

    /// filter on basic jets
    typedef ObjectCountFilter<reco::BasicJetCollection, 
			      AndSelector<EtMinSelector<reco::BasicJet>,
					  EtaRangeSelector<reco::BasicJet> > 
                             > EtaEtMinBasicJetCountFilter;

    /// filter on generator jets
    typedef ObjectCountFilter<reco::GenJetCollection, 
			      AndSelector<EtMinSelector<reco::GenJet>,
					  EtaRangeSelector<reco::GenJet> > 
                             > EtaEtMinGenJetCountFilter;

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

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( TrackCountFilter );
DEFINE_ANOTHER_FWK_MODULE( ElectronCountFilter );
DEFINE_ANOTHER_FWK_MODULE( PhotonCountFilter );
DEFINE_ANOTHER_FWK_MODULE( MuonCountFilter );
DEFINE_ANOTHER_FWK_MODULE( CaloJetCountFilter );
DEFINE_ANOTHER_FWK_MODULE( BasicJetCountFilter );
DEFINE_ANOTHER_FWK_MODULE( GenJetCountFilter );
DEFINE_ANOTHER_FWK_MODULE( PtMinTrackCountFilter );
DEFINE_ANOTHER_FWK_MODULE( PtMinElectronCountFilter );
DEFINE_ANOTHER_FWK_MODULE( PtMinPhotonCountFilter );
DEFINE_ANOTHER_FWK_MODULE( PtMinMuonCountFilter );
DEFINE_ANOTHER_FWK_MODULE( EtMinCaloJetCountFilter );
DEFINE_ANOTHER_FWK_MODULE( EtMinBasicJetCountFilter );
DEFINE_ANOTHER_FWK_MODULE( EtMinGenJetCountFilter );
DEFINE_ANOTHER_FWK_MODULE( EtaEtMinCaloJetCountFilter );
DEFINE_ANOTHER_FWK_MODULE( EtaEtMinBasicJetCountFilter );
DEFINE_ANOTHER_FWK_MODULE( EtaEtMinGenJetCountFilter );

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
