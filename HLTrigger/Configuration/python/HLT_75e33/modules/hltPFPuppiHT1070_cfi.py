import FWCore.ParameterSet.Config as cms

hltPFPuppiHT1070 = cms.EDFilter("HLTHtMhtFilter",
    htLabels = cms.VInputTag("hltPFPuppiHT"),
    meffSlope = cms.vdouble(1.0),
    mhtLabels = cms.VInputTag("hltPFPuppiHT"),
    minHt = cms.vdouble(1070.0),
    minMeff = cms.vdouble(0.0),
    minMht = cms.vdouble(0.0),
    saveTags = cms.bool(True)
)
