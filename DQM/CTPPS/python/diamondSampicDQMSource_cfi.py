import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
diamondSampicDQMSourceOnline = DQMEDAnalyzer('DiamondSampicDQMSource',
    tagDigi = cms.untracked.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagFEDInfo = cms.untracked.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagRecHits = cms.untracked.InputTag("totemTimingRecHits"),
    tagTracks = cms.untracked.InputTag("diamondSampicLocalTracks"),
    tagLocalTrack = cms.untracked.InputTag("totemRPLocalTrackFitter"),

    samplesForNoise = cms.untracked.uint32(6),

    verbosity = cms.untracked.uint32(10),
    plotOnline=cms.untracked.bool(True)
)

diamondSampicDQMSourceOffline = DQMEDAnalyzer('DiamondSampicDQMSource',
    tagDigi = cms.untracked.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagFEDInfo = cms.untracked.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagRecHits = cms.untracked.InputTag("totemTimingRecHits"),
    tagTracks = cms.untracked.InputTag("diamondSampicLocalTracks"),
    tagLocalTrack = cms.untracked.InputTag("totemRPLocalTrackFitter"),

    samplesForNoise = cms.untracked.uint32(6),

    perLSsaving = cms.untracked.bool(False), #driven by DQMServices/Core/python/DQMStore_cfi.py

    verbosity = cms.untracked.uint32(10),
    plotOnline=cms.untracked.bool(False)
)
