import FWCore.ParameterSet.Config as cms

PixelQuadrupletGenerator = cms.PSet(
    ComponentName = cms.string("PixelQuadrupletGenerator"),
    extraHitRZtolerance = cms.double(0.1),
    extraHitRPhitolerance = cms.double(0.1),
    extraPhiTolerance = cms.PSet(
        pt1    = cms.double(0.1) , pt2    = cms.double(0.1),
        value1 = cms.double(999.), value2 = cms.double(0.15),
        enabled = cms.bool(False),
    ),
    maxChi2 = cms.PSet(
        pt1    = cms.double(0.2), pt2    = cms.double(1.5),
        value1 = cms.double(500), value2 = cms.double(50),
        enabled = cms.bool(True),
    ),
    fitFastCircle = cms.bool(False),
    fitFastCircleChi2Cut = cms.bool(False),
    useBendingCorrection = cms.bool(False),
)
