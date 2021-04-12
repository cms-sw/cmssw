import FWCore.ParameterSet.Config as cms

hltIter2Phase2L3FromL1TkMuonPixelHitTriplets = cms.EDProducer("CAHitTripletEDProducer",
    CAHardPtCut = cms.double(0.3),
    CAPhiCut = cms.double(0.1),
    CAThetaCut = cms.double(0.015),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    doublets = cms.InputTag("hltIter2Phase2L3FromL1TkMuonPixelHitDoublets"),
    extraHitRPhitolerance = cms.double(0.032),
    maxChi2 = cms.PSet(
        enabled = cms.bool(True),
        pt1 = cms.double(0.8),
        pt2 = cms.double(8.0),
        value1 = cms.double(100.0),
        value2 = cms.double(6.0)
    ),
    useBendingCorrection = cms.bool(True)
)
