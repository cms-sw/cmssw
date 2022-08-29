import FWCore.ParameterSet.Config as cms

l1tTkIsoEleSingle28Filter = cms.EDFilter("L1TTkEleFilter",
    ApplyQual1 = cms.bool(True),
    ApplyQual2 = cms.bool(True),
    EtaBinsForIsolation = cms.vdouble(0.0, 1.479, 2.4),
    MaxEta = cms.double(2.4),
    MinEta = cms.double(-2.4),
    MinN = cms.int32(1),
    MinPt = cms.double(28.0),
    Qual1IsMask = cms.bool(True),
    Qual2IsMask = cms.bool(False),
    Quality1 = cms.int32(2),
    Quality2 = cms.int32(5),
    Scalings = cms.PSet(
        barrel = cms.vdouble(0.434262, 1.20586, 0.0),
        endcap = cms.vdouble(0.266186, 1.25976, 0.0)
    ),
    TrkIsolation = cms.vdouble(0.12, 0.2),
    inputTag1 = cms.InputTag("l1tLayer1EG","L1TkEleEB"),
    inputTag2 = cms.InputTag("l1tLayer1EG","L1TkEleEE"),
    saveTags = cms.bool(True)
)
