import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
totemTimingDQMSource = DQMEDAnalyzer('TotemTimingDQMSource',
    tagDigi = cms.untracked.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagFEDInfo = cms.untracked.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagRecHits = cms.untracked.InputTag("totemTimingRecHits"),
    # tagTracks = cms.untracked.InputTag("totemTimingLocalTracks"),
    tagLocalTrack = cms.untracked.InputTag("totemRPLocalTrackFitter"),

    minimumStripAngleForTomography = cms.double(0),
    maximumStripAngleForTomography = cms.double(1),
    samplesForNoise = cms.untracked.uint32(6),

    perLSsaving = cms.untracked.bool(False), #driven by DQMServices/Core/python/DQMStore_cfi.py

    verbosity = cms.untracked.uint32(10),
)
