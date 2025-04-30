import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using MinBias events
OutALCARECOTkAlHLTTracks_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlHLTTracks')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlHLTTracks_*_*',
        'keep recoTrackExtras_ALCARECOTkAlHLTTracks_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlHLTTracks_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlHLTTracks_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlHLTTracks_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_hltVerticesPFFilter_*_*',
        'keep *_hltOnlineBeamSpot_*_*')
)

OutALCARECOTkAlHLTTracks = OutALCARECOTkAlHLTTracks_noDrop.clone()
OutALCARECOTkAlHLTTracks.outputCommands.insert(0, "drop *")
-- dummy change --
