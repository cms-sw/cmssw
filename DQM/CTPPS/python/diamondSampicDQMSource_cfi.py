import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
diamondSampicDQMSourceOnline = DQMEDAnalyzer('DiamondSampicDQMSource',
    tagDigi = cms.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagFEDInfo = cms.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagRecHits = cms.InputTag("totemTimingRecHits"),
    tagTracks = cms.InputTag("diamondSampicLocalTracks"),
    tagLocalTrack = cms.InputTag("totemRPLocalTrackFitter"),

    samplesForNoise = cms.untracked.uint32(6),

    verbosity = cms.untracked.uint32(10),
    plotOnline=cms.untracked.bool(True)
)

diamondSampicDQMSourceOffline = DQMEDAnalyzer('DiamondSampicDQMSource',
    tagDigi = cms.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagFEDInfo = cms.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagRecHits = cms.InputTag("totemTimingRecHits"),
    tagTracks = cms.InputTag("diamondSampicLocalTracks"),
    tagLocalTrack = cms.InputTag("totemRPLocalTrackFitter"),

    samplesForNoise = cms.untracked.uint32(6),

    verbosity = cms.untracked.uint32(10),
    plotOnline=cms.untracked.bool(False)
)
