import FWCore.ParameterSet.Config as cms

selectedLayer1METs = cms.EDFilter("PATMETSelector",
    src = cms.InputTag("allLayer1METs"),
    cut = cms.string('et > 0.')
)


