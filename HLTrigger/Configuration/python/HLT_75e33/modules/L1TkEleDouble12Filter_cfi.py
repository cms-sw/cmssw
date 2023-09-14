import FWCore.ParameterSet.Config as cms

l1tTkEleDouble12Filter = cms.EDFilter("L1TTkEleFilter",
    ApplyQual1 = cms.bool(True),
    ApplyQual2 = cms.bool(True),
    EtaBinsForIsolation = cms.vdouble(0.0, 1.479, 2.4),
    MinAbsEta1 = cms.double(0.0),
    MaxAbsEta1 = cms.double(1.479),
    MinAbsEta2 = cms.double(1.479),
    MaxAbsEta2 = cms.double(2.4),
    MinN = cms.int32(2),
    MinPt = cms.double(12.0),
    Qual1IsMask = cms.bool(True),
    Qual2IsMask = cms.bool(True),
    Quality1 = cms.int32(2),
    Quality2 = cms.int32(2),
    Scalings = cms.PSet(
        barrel = cms.vdouble(0.805095, 1.18336, 0.0),
        endcap = cms.vdouble(0.453144, 1.26205, 0.0)
    ),
    TrkIsolation = cms.vdouble(99999.0, 99999.0),
    inputTag1 = cms.InputTag("l1tLayer1EG","L1TkEleEB"),
    inputTag2 = cms.InputTag("l1tLayer1EG","L1TkEleEE"),
    saveTags = cms.bool(True)
)
