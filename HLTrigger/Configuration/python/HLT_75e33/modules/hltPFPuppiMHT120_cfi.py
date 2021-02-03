import FWCore.ParameterSet.Config as cms

hltPFPuppiMHT120 = cms.EDFilter("HLTMhtFilter",
    mhtLabels = cms.VInputTag("hltPFPuppiMHT"),
    minMht = cms.vdouble(120.0),
    saveTags = cms.bool(True)
)
