import FWCore.ParameterSet.Config as cms

hltPFPuppiCentralJetsQuad30HT200MaxEta2p4 = cms.EDFilter("HLTHtMhtFilter",
    htLabels = cms.VInputTag("hltHtMhtPFPuppiCentralJetsQuadC30MaxEta2p4"),
    meffSlope = cms.vdouble(1.0),
    mhtLabels = cms.VInputTag("hltHtMhtPFPuppiCentralJetsQuadC30MaxEta2p4"),
    minHt = cms.vdouble(200.0),
    minMeff = cms.vdouble(0.0),
    minMht = cms.vdouble(0.0),
    saveTags = cms.bool(True)
)
