
full_title = "RecoMuon collections (in RECO and AOD)"

full = {
    
      # Correction needed, because not matched with Event Content    
    '0':['muonsFromCosmics','reco::MuonCollection','Similar to `muons` but reconstructed by dedicated cosmic-muon reconstructor (2-leg option)'],
    '1':['muonsWithSET','reco::MuonCollection','Similar to `muons` but reconstructed by alternative SET algorithm for standalone muons'],
    '2':['standAloneMuons:UpdatedAtVtx','reco::TrackCollection','Standalone muon tracks without extra and hits, with constraint at the beam spot'],
    '3':['tevMuons:default','reco::TrackCollection','Global muon tracks using the global-muon reconstruction algorithm with one refit'],
    '4':['tevMuons:firstHit','reco::TrackCollection','Global muon tracks using all tracker hits plus hits from the first muon station that has them'],
    '5':['tevMuons:picky','reco::TrackCollection','Global muon tracks using all tracker hits plus hits from the muon stations which do not contain electromagnetic showers'],
    '6':['tevMuons:dyt','reco::TrackCollection','Global muon tracks reconstructed by the DYT algorithm for high-p<sub>T</sub> muons'],
    '7':['muIsoDepositTk','reco::IsoDepositMap','Map of IsoDeposits for each muon calculated using general tracks'],
    '8':['muIsoDepositCalByAssociatorTowers','reco::IsoDepositMap','Map of IsoDeposits for each muon calculated using calorimeter towers. Three instances are created (corresponding to the calo-tower component): <code>ecal</code>, <code>hcal</code>, and <code>ho</code>'],
    '9':['muIsoDepositJets','reco::IsoDepositMap','Map of IsoDeposits for each muon calculated using jets (sisCone5CaloJets)'],
    '10':['tevMuons:default','reco::TrackToTrackMap','Map associating tracks in the globalMuons collection to tracks in the tevMuons:default collection'],
    '11':['tevMuons:firstHit','reco::TrackToTrackMap','Map associating tracks in the globalMuons collection to tracks in the tevMuons:firstHit collection'],
    '12':['tevMuons:picky','reco::TrackToTrackMap','Map associating tracks in the globalMuons collection to tracks in the tevMuons:picky collection'],
    '13':['tevMuons:dyt','reco::TrackToTrackMap','Map associating tracks in the globalMuons collection to tracks in the tevMuons:dyt collection'],
    '14':['muid*','edm::ValueMap<boolean>','Output of the muon selectors defined in DataFormats/MuonReco/interface/MuonSelectors.h'],
    '15':['MuonShowerInformation','edm::ValueMap<reco::MuonShower>','Muon shower information, defined in DataFormats/MuonReco/interface/MuonShower.h'],
    '16':['muons','edm::ValueMap<reco::MuonTimeExtra>','Muon timing information for DT, CSC, and combined, defined in DataFormats/MuonReco/interface/MuonTimeExtra.h'],
    '17':['cosmicsVeto','edm::ValueMap<unsigned int>','Index of the partner track found in the opposite hemisphere, used by the cosmic-muon identifier'],
    '18':['cosmicsVeto','edm::ValueMap<reco::MuonCosmicCompatibility>','Information used by the cosmic-muon identifier, defined in DataFormats/MuonReco/interface/MuonCosmicCompatibility.h']

}

reco_title = "RecoMuon collections (in RECO only)"

reco = {
    '0':['MuonSeed', 'TrajectorySeed', 'Seeds for standalone muon reconstruction'] ,
    '1':['ancientMuonSeed', '*', 'No documentation'] ,
    '2':['mergedStandAloneMuonSeeds', '*', 'No documentation'] ,
    '3':['globalMuons', 'TrackingRecHitsOwned', 'No documentation'] ,
    '4':['tevMuons', 'TrackingRecHitsOwned', 'No documentation'] ,
    '5':['calomuons', 'recoCaloMuons', 'Tracks with energy depositions in the calorimeters consistent with those of a muon, but which failed all other muon reconstruction algorithms'] ,
    # Cosmics
    '6':['CosmicMuonSeed', 'TrajectorySeed', 'Seeds for dedicated cosmic reconstructor of standalone muons'] ,
    '7':['cosmicMuons', 'recoTrackExtras', 'No documentation'] ,
    '8':['cosmicMuons', 'TrackingRecHitsOwned', 'No documentation'] ,
    '9':['globalCosmicMuons', 'recoTrackExtras', 'No documentation'] ,
    '10':['globalCosmicMuons', 'TrackingRecHitsOwned', 'No documentation'] ,
    '11':['cosmicMuons1Leg', 'recoTrackExtras', 'No documentation'] ,
    '12':['cosmicMuons1Leg', 'TrackingRecHitsOwned', 'No documentation'] ,
    '13':['globalCosmicMuons1Leg', 'recoTrackExtras', 'No documentation'] ,
    '14':['globalCosmicMuons1Leg', 'TrackingRecHitsOwned', 'No documentation'] ,
    '15':['cosmicsVetoTracks', 'recoTracks', 'No documentation'] ,
    # SET muons
    '16':['SETMuonSeed', '*', 'No documentation'] ,
    '17':['standAloneSETMuons', 'recoTracks', 'No documentation'] ,
    '18':['standAloneSETMuons', 'recoTrackExtras', 'No documentation'] ,
    '19':['standAloneSETMuons', 'TrackingRecHitsOwned', 'No documentation'] ,
    '20':['globalSETMuons', 'recoTracks', 'No documentation'] ,
    '21':['globalSETMuons', 'recoTrackExtras', 'No documentation'] ,
    '22':['globalSETMuons', 'TrackingRecHitsOwned', 'No documentation'],
    
    '23':['standAloneSETMuons','reco::TrackCollection','Similar to standAloneMuons but produced by alternative SET algorithm'],
    '24':['globalSETMuons','reco::TrackCollection','Similar to globalMuons but produced by alternative SET algorithm'] 
}

aod_title = "RecoMuon collections (in AOD only)"

aod = {
    '0':['muons', 'reco::MuonCollection', 'Muons built using tracker-muon, standalone-muon and global-muon reconstruction algorithms with muon id and other information (energy deposits, isolation information, etc.)'] ,
    '1':['*', '*', 'No documentation'] ,
    '2':['standAloneMuons', 'recoTracks', 'Standalone muon tracks without extra and hits'] ,
    '3':['standAloneMuons', 'recoTrackExtras', 'No documentation'] ,
    '4':['standAloneMuons', 'TrackingRecHitsOwned', 'No documentation'] ,
    '5':['globalMuons', 'recoTracks', 'Global muon tracks without extra and hits'] ,
    '6':['globalMuons', 'recoTrackExtras', 'No documentation'] ,
    '7':['tevMuons', 'recoTracks', 'No documentation'] ,
    '8':['tevMuons', 'recoTrackExtras', 'No documentation'] ,
    '9':['generalTracks', 'recoTracks', 'No documentation'] ,
    '10':['tevMuons', 'recoTracksToOnerecoTracksAssociation', 'No documentation'] ,
    # cosmics
    '11':['cosmicMuons', 'recoTracks', 'Standalone muon tracks reconstructed by dedicated cosmic-muon reconstructor (2-leg option)'] ,
    '12':['globalCosmicMuons', 'recoTracks', 'Global muon tracks reconstructed by dedicated cosmic-muon reconstructor (2-leg option)'] ,
    '13':['muonsFromCosmics', 'recoMuons', 'No documentation'] ,
    # cosmics 1 leg
    '14':['cosmicMuons1Leg', 'recoTracks', 'Standalone muon tracks reconstructed by dedicated cosmic-muon reconstructor (1-leg option)'] ,
    '15':['globalCosmicMuons1Leg', 'recoTracks', 'Global muon tracks reconstructed by dedicated cosmic-muon reconstructor (1-leg option)'] ,
    '16':['muonsFromCosmics1Leg', 'recoMuons', 'Similar to `muons` but reconstructed by dedicated cosmic-muon reconstructor (1-leg option)'] ,
    # additional tracks
    '17':['refittedStandAloneMuons', 'recoTracks', 'No documentation'] ,
    '18':['refittedStandAloneMuons', 'recoTrackExtras', 'No documentation'] ,
    '19':['refittedStandAloneMuons', 'TrackingRecHitsOwned', 'No documentation'] 
}

