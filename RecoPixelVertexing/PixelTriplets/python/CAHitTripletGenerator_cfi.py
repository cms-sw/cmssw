import FWCore.ParameterSet.Config as cms

CAHitTripletGenerator = cms.PSet(
    ComponentName = cms.string("CAHitTripletGenerator"),
    extraHitRPhitolerance = cms.double(0.06),
    maxChi2 = cms.PSet(
        pt1    = cms.double(0.8), pt2    = cms.double(2),
        value1 = cms.double(50), value2 = cms.double(8),
        enabled = cms.bool(True),
    ),
    useBendingCorrection = cms.bool(False),
    CAThetaCut = cms.double(0.00125),
    CAPhiCut = cms.double(1),
)


