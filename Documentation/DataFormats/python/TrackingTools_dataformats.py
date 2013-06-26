full_title = "TrackingTools collections (in RECO and AOD)"

full = {
    '0':['CkfElectronCandidates', '*', 'No documentation'] ,
    '1':['GsfGlobalElectronTest', '*', 'No documentation'] ,
    '2':['electronMergedSeeds', '*', 'No documentation'] ,
    '3':['electronGsfTracks', '*', 'No documentation'] 
}

reco_title = "TrackingTools collections (in RECO only)"

reco = {
    '0':['CkfElectronCandidates', '*', 'No documentation'] ,
    '1':['GsfGlobalElectronTest', '*', 'No documentation'] ,
    '2':['electronMergedSeeds', '*', 'No documentation'] ,
    '3':['electronGsfTracks', 'recoGsfTracks', 'No documentation'] ,
    '4':['electronGsfTracks', 'recoGsfTrackExtras', 'No documentation'] ,
    '5':['electronGsfTracks', 'recoTrackExtras', 'No documentation'] 
}

aod_title = "TrackingTools collections (in AOD only)"

aod = {
    '0':['GsfGlobalElectronTest', 'recoTracks', 'No documentation'] ,
    '1':['electronGsfTracks', 'recoGsfTracks', 'No documentation'] 
}