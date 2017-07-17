import FWCore.ParameterSet.Config as cms

CAHitQuadrupletGenerator = cms.PSet(
    ComponentName = cms.string("CAHitQuadrupletGenerator"),
    SeedingLayers = cms.InputTag("seedingLayersEDProducer"),
    extraHitRPhitolerance = cms.double(0.1),
    maxChi2 = cms.PSet(
        pt1    = cms.double(0.2), pt2    = cms.double(1.5),
        value1 = cms.double(500), value2 = cms.double(50),
        enabled = cms.bool(True),
    ),
    fitFastCircle = cms.bool(False),
    fitFastCircleChi2Cut = cms.bool(False),
    useBendingCorrection = cms.bool(False),
    CAThetaCut = cms.double(0.00125),
    CAPhiCut = cms.double(10),
    CAHardPtCut = cms.double(0),
    CAOnlyOneLastHitPerLayerFilter= cms.bool(False)
)
