import FWCore.ParameterSet.Config as cms

l1tTkEmDouble12Filter = cms.EDFilter("L1TTkEmFilter",
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
    Quality2 = cms.int32(4),
    Scalings = cms.PSet(
        barrel = cms.vdouble(2.6604, 1.06077, 0.0),
        endcap = cms.vdouble(3.17445, 1.13219, 0.0)
    ),
    TrkIsolation = cms.vdouble(99999.0, 99999.0),
    inputTag1 = cms.InputTag("l1tLayer1EG","L1TkEmEB"),
    inputTag2 = cms.InputTag("l1tLayer1EG","L1TkEmEE"),
    saveTags = cms.bool(True)
)
