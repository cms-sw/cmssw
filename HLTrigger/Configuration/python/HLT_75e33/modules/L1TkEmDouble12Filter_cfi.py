import FWCore.ParameterSet.Config as cms

L1TkEmDouble12Filter = cms.EDFilter("L1TTkEmFilter",
    ApplyQual1 = cms.bool(True),
    ApplyQual2 = cms.bool(True),
    EtaBinsForIsolation = cms.vdouble(0.0, 1.479, 2.4),
    MaxEta = cms.double(2.4),
    MinEta = cms.double(-2.4),
    MinN = cms.int32(2),
    MinPt = cms.double(12.0),
    Qual1IsMask = cms.bool(True),
    Qual2IsMask = cms.bool(False),
    Quality1 = cms.int32(2),
    Quality2 = cms.int32(5),
    Scalings = cms.PSet(
        barrel = cms.vdouble(2.6604, 1.06077, 0.0),
        endcap = cms.vdouble(3.17445, 1.13219, 0.0)
    ),
    TrkIsolation = cms.vdouble(99999.0, 99999.0),
    inputTag1 = cms.InputTag("hltL1TkPhotonsCrystal","EG"),
    inputTag2 = cms.InputTag("hltL1TkPhotonsHGC","EG"),
    saveTags = cms.bool(True)
)
