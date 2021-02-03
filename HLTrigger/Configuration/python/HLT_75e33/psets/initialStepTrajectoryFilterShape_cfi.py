import FWCore.ParameterSet.Config as cms

initialStepTrajectoryFilterShape = cms.PSet(
    ComponentType = cms.string('StripSubClusterShapeTrajectoryFilter'),
    layerMask = cms.PSet(
        TEC = cms.bool(False),
        TIB = cms.vuint32(1, 2),
        TID = cms.vuint32(1, 2),
        TOB = cms.bool(False)
    ),
    maxNSat = cms.uint32(3),
    maxTrimmedSizeDiffNeg = cms.double(1.0),
    maxTrimmedSizeDiffPos = cms.double(0.7),
    seedCutMIPs = cms.double(0.35),
    seedCutSN = cms.double(7.0),
    subclusterCutMIPs = cms.double(0.45),
    subclusterCutSN = cms.double(12.0),
    subclusterWindow = cms.double(0.7),
    trimMaxADC = cms.double(30.0),
    trimMaxFracNeigh = cms.double(0.25),
    trimMaxFracTotal = cms.double(0.15)
)