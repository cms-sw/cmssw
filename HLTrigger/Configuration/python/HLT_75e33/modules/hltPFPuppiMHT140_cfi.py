import FWCore.ParameterSet.Config as cms

hltPFPuppiMHT140 = cms.EDFilter("HLTMhtFilter",
    mhtLabels = cms.VInputTag("hltPFPuppiMHT"),
    minMht = cms.vdouble(140.0),
    saveTags = cms.bool(True)
)
