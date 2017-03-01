import FWCore.ParameterSet.Config as cms

totemRPLocalTrackFitter = cms.EDProducer("TotemRPLocalTrackFitter",
    verbosity = cms.int32(0),

    tagUVPattern = cms.InputTag("totemRPUVPatternFinder")
)
