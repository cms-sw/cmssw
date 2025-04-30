import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using ZMuMu events
OutALCARECOTkAlHLTTracksZMuMu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlHLTTracksZMuMu')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlHLTTracksZMuMu_*_*',
        'keep recoTrackExtras_ALCARECOTkAlHLTTracksZMuMu_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlHLTTracksZMuMu_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlHLTTracksZMuMu_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlHLTTracksZMuMu_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep *_TriggerResults_*_*',
	'keep *_hltVerticesPFFilter_*_*',
        'keep *_hltOnlineBeamSpot_*_*'
    )
)

OutALCARECOTkAlHLTTracksZMuMu = OutALCARECOTkAlHLTTracksZMuMu_noDrop.clone()
OutALCARECOTkAlHLTTracksZMuMu.outputCommands.insert(0, "drop *")

-- dummy change --
