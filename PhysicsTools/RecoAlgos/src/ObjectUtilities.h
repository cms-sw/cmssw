#ifndef RecoAlgos_ObjectUtilities_h
#define RecoAlgos_ObjectUtilities_h
#include "PhysicsTools/UtilAlgos/interface/CollectionRecoverer.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

namespace reco {
  namespace modules {
    /// Framework module to merge an arbitray number of reco::TrackCollection
    typedef Merger<reco::TrackCollection> TrackMerger;

    /// Framework module to merge an arbitray number of reco::MuonCollection
    typedef Merger<reco::MuonCollection> MuonMerger;

    /// Framework module to merge an arbitray number of reco::ElectronCollection
    typedef Merger<reco::ElectronCollection> ElectronMerger;

    /// Framework module to merge an arbitray number of reco::PhotonCollection
    typedef Merger<reco::PhotonCollection> PhotonMerger;

    /// Recover a collection of reco::Track
    typedef CollectionRecoverer<reco::TrackCollection> TrackRecoverer; 
  }
}

#endif
