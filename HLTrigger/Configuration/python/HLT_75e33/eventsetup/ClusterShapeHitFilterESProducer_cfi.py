import FWCore.ParameterSet.Config as cms

ClusterShapeHitFilterESProducer = cms.ESProducer("ClusterShapeHitFilterESProducer",
    ComponentName = cms.string('ClusterShapeHitFilter'),
    PixelShapeFile = cms.string('RecoTracker/PixelLowPtUtilities/data/ITShapePhase2_all.par'),
    PixelShapeFileL1 = cms.string('RecoTracker/PixelLowPtUtilities/data/ITShapePhase2_all.par'),
    appendToDataLabel = cms.string(''),
    clusterChargeCut = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutNone')
    ),
    doPixelShapeCut = cms.bool(True),
    doStripShapeCut = cms.bool(True),
    isPhase2 = cms.bool(True)
)
