'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoMuon collections (in RECO and AOD)",
    "data": [
     {
      "instance": "tevMuons:firstHit",
      "container": "reco::TrackToTrackMap",
      "desc": "Map associating tracks in the globalMuons collection to tracks in the tevMuons:firstHit collection"
     },
     {
      "instance": "tevMuons:default",
      "container": "reco::TrackToTrackMap",
      "desc": "Map associating tracks in the globalMuons collection to tracks in the tevMuons:default collection"
     },
     {
      "instance": "tevMuons:dyt",
      "container": "reco::TrackToTrackMap",
      "desc": "Map associating tracks in the globalMuons collection to tracks in the tevMuons:dyt collection"
     },
     {
      "instance": "tevMuons:picky",
      "container": "reco::TrackToTrackMap",
      "desc": "Map associating tracks in the globalMuons collection to tracks in the tevMuons:picky collection"
     },
     {
      "instance": "MuonShowerInformation",
      "container": "edm::ValueMap<reco::MuonShower>",
      "desc": "Muon shower information, defined in DataFormats/MuonReco/interface/MuonShower.h"
     },
     {
      "instance": "muid*",
      "container": "edm::ValueMap<boolean>",
      "desc": "Output of the muon selectors defined in DataFormats/MuonReco/interface/MuonSelectors.h"
     },
     {
      "instance": "cosmicsVeto",
      "container": "edm::ValueMap<unsigned int>",
      "desc": "Index of the partner track found in the opposite hemisphere, used by the cosmic-muon identifier"
     },
     {
      "instance": "muons",
      "container": "edm::ValueMap<reco::MuonTimeExtra>",
      "desc": "Muon timing information for DT, CSC, and combined, defined in DataFormats/MuonReco/interface/MuonTimeExtra.h"
     },
     {
      "instance": "cosmicsVeto",
      "container": "edm::ValueMap<reco::MuonCosmicCompatibility>",
      "desc": "Information used by the cosmic-muon identifier, defined in DataFormats/MuonReco/interface/MuonCosmicCompatibility.h"
     },
     {
      "instance": "muonsWithSET",
      "container": "reco::MuonCollection",
      "desc": "Similar to `muons` but reconstructed by alternative SET algorithm for standalone muons"
     },
     {
      "instance": "muonsFromCosmics",
      "container": "reco::MuonCollection",
      "desc": "Similar to `muons` but reconstructed by dedicated cosmic-muon reconstructor (2-leg option)"
     },
     {
      "instance": "tevMuons:default",
      "container": "reco::TrackCollection",
      "desc": "Global muon tracks using the global-muon reconstruction algorithm with one refit"
     },
     {
      "instance": "standAloneMuons:UpdatedAtVtx",
      "container": "reco::TrackCollection",
      "desc": "Standalone muon tracks without extra and hits, with constraint at the beam spot"
     },
     {
      "instance": "tevMuons:picky",
      "container": "reco::TrackCollection",
      "desc": "Global muon tracks using all tracker hits plus hits from the muon stations which do not contain electromagnetic showers"
     },
     {
      "instance": "tevMuons:firstHit",
      "container": "reco::TrackCollection",
      "desc": "Global muon tracks using all tracker hits plus hits from the first muon station that has them"
     },
     {
      "instance": "muIsoDepositTk",
      "container": "reco::IsoDepositMap",
      "desc": "Map of IsoDeposits for each muon calculated using general tracks"
     },
     {
      "instance": "tevMuons:dyt",
      "container": "reco::TrackCollection",
      "desc": "Global muon tracks reconstructed by the DYT algorithm for high-p<sub>T</sub> muons"
     },
     {
      "instance": "muIsoDepositJets",
      "container": "reco::IsoDepositMap",
      "desc": "Map of IsoDeposits for each muon calculated using jets (sisCone5CaloJets)"
     },
     {
      "instance": "muIsoDepositCalByAssociatorTowers",
      "container": "reco::IsoDepositMap",
      "desc": "Map of IsoDeposits for each muon calculated using calorimeter towers. Three instances are created (corresponding to the calo-tower component): <code>ecal</code>, <code>hcal</code>, and <code>ho</code>"
     }
    ]
  },
  "aod": {
    "title": "RecoMuon collections (in AOD only)",
    "data": [
     {
      "instance": "cosmicMuons",
      "container": "recoTracks",
      "desc": "Standalone muon tracks reconstructed by dedicated cosmic-muon reconstructor (2-leg option)"
     },
     {
      "instance": "tevMuons",
      "container": "recoTracksToOnerecoTracksAssociation",
      "desc": "No documentation"
     },
     {
      "instance": "muonsFromCosmics",
      "container": "recoMuons",
      "desc": "No documentation"
     },
     {
      "instance": "globalCosmicMuons",
      "container": "recoTracks",
      "desc": "Global muon tracks reconstructed by dedicated cosmic-muon reconstructor (2-leg option)"
     },
     {
      "instance": "globalCosmicMuons1Leg",
      "container": "recoTracks",
      "desc": "Global muon tracks reconstructed by dedicated cosmic-muon reconstructor (1-leg option)"
     },
     {
      "instance": "cosmicMuons1Leg",
      "container": "recoTracks",
      "desc": "Standalone muon tracks reconstructed by dedicated cosmic-muon reconstructor (1-leg option)"
     },
     {
      "instance": "refittedStandAloneMuons",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "muonsFromCosmics1Leg",
      "container": "recoMuons",
      "desc": "Similar to `muons` but reconstructed by dedicated cosmic-muon reconstructor (1-leg option)"
     },
     {
      "instance": "refittedStandAloneMuons",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "refittedStandAloneMuons",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "muons",
      "container": "reco::MuonCollection",
      "desc": "Muons built using tracker-muon, standalone-muon and global-muon reconstruction algorithms with muon id and other information (energy deposits, isolation information, etc.)"
     },
     {
      "instance": "standAloneMuons",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "standAloneMuons",
      "container": "recoTracks",
      "desc": "Standalone muon tracks without extra and hits"
     },
     {
      "instance": "globalMuons",
      "container": "recoTracks",
      "desc": "Global muon tracks without extra and hits"
     },
     {
      "instance": "standAloneMuons",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "tevMuons",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "globalMuons",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "generalTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "tevMuons",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "RecoMuon collections (in RECO only)",
    "data": [
     {
      "instance": "globalSETMuons",
      "container": "reco::TrackCollection",
      "desc": "Similar to globalMuons but produced by alternative SET algorithm"
     },
     {
      "instance": "globalSETMuons",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "globalSETMuons",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "globalSETMuons",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "standAloneSETMuons",
      "container": "reco::TrackCollection",
      "desc": "Similar to standAloneMuons but produced by alternative SET algorithm"
     },
     {
      "instance": "ancientMuonSeed",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "MuonSeed",
      "container": "TrajectorySeed",
      "desc": "Seeds for standalone muon reconstruction"
     },
     {
      "instance": "globalMuons",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "mergedStandAloneMuonSeeds",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "calomuons",
      "container": "recoCaloMuons",
      "desc": "Tracks with energy depositions in the calorimeters consistent with those of a muon, but which failed all other muon reconstruction algorithms"
     },
     {
      "instance": "tevMuons",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "cosmicMuons",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "CosmicMuonSeed",
      "container": "TrajectorySeed",
      "desc": "Seeds for dedicated cosmic reconstructor of standalone muons"
     },
     {
      "instance": "globalCosmicMuons",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "cosmicMuons",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "cosmicMuons1Leg",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "globalCosmicMuons",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "globalCosmicMuons1Leg",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "cosmicMuons1Leg",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "cosmicsVetoTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "globalCosmicMuons1Leg",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "standAloneSETMuons",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "SETMuonSeed",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "standAloneSETMuons",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "standAloneSETMuons",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     }
    ]
  }
}
