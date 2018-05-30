import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
totemTimingDQMSource = DQMEDAnalyzer('TotemTimingDQMSource',
    tagDigi = cms.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagFEDInfo = cms.InputTag("totemTimingRawToDigi", "TotemTiming"),
    tagRecHits = cms.InputTag("totemTimingRecHits"),
    # tagTracks = cms.InputTag("totemTimingLocalTracks"),
    tagLocalTrack = cms.InputTag("totemRPLocalTrackFitter"),

    minimumStripAngleForTomography = cms.double(0),
    maximumStripAngleForTomography = cms.double(1),
    samplesForNoise = cms.untracked.uint32(6),

    verbosity = cms.untracked.uint32(10),
)