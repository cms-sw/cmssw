import FWCore.ParameterSet.Config as cms

pixelLessStepClusterShapeHitFilter = cms.ESProducer("ClusterShapeHitFilterESProducer",
    ComponentName = cms.string('pixelLessStepClusterShapeHitFilter'),
    PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/ITShapePhase2_all.par'),
    PixelShapeFileL1 = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/ITShapePhase2_all.par'),
    clusterChargeCut = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutTight')
    ),
    doStripShapeCut = cms.bool(False)
)
