#ifndef RecoAlgos_ObjectCountFilters_h
#define RecoAlgos_ObjectCountFilters_h
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/Utilities/interface/AndSelector.h"
#include "PhysicsTools/Utilities/interface/OrSelector.h"
#include "PhysicsTools/Utilities/interface/PtMinSelector.h"
#include "PhysicsTools/Utilities/interface/EtMinSelector.h"
#include "PhysicsTools/Utilities/interface/EtaRangeSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

namespace reco {
  namespace modules {
    /// filter on track number
    typedef ObjectCountFilter<
              reco::TrackCollection
            > TrackCountFilter;

    /// filter on electron number
    typedef ObjectCountFilter<
              reco::ElectronCollection
            > ElectronCountFilter;

    /// filter on photon number
    typedef ObjectCountFilter<
              reco::PhotonCollection
            > PhotonCountFilter;

    /// filter on muon number
    typedef ObjectCountFilter<
              reco::MuonCollection
            > MuonCountFilter;

    /// filter on calo jet number
    typedef ObjectCountFilter<
              reco::CaloJetCollection
            > CaloJetCountFilter;

    /// filter on track number
    typedef ObjectCountFilter<
              reco::TrackCollection, 
              PtMinSelector<reco::Track> 
            > PtMinTrackCountFilter;

    /// filter on electron number
    typedef ObjectCountFilter<
              reco::ElectronCollection, 
              PtMinSelector<reco::Electron> 
            > PtMinElectronCountFilter;

    /// filter on photon number
    typedef ObjectCountFilter<
              reco::PhotonCollection, 
              PtMinSelector<reco::Photon> 
            > PtMinPhotonCountFilter;

    /// filter on muon number
    typedef ObjectCountFilter<
              reco::MuonCollection, 
              PtMinSelector<reco::Muon> 
            > PtMinMuonCountFilter;

    /// filter on calo jets
    typedef ObjectCountFilter<
              reco::CaloJetCollection, 
              EtMinSelector<reco::CaloJet> 
            > EtMinCaloJetCountFilter;

    /// filter on calo jets
    typedef ObjectCountFilter<
              reco::CaloJetCollection, 
	      AndSelector<
                EtMinSelector<reco::CaloJet>,
	       EtaRangeSelector<reco::CaloJet> 
              > 
            > EtaEtMinCaloJetCountFilter;

  }
}
#endif
