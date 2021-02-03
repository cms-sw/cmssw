import FWCore.ParameterSet.Config as cms

mixedTripletStepClusterShapeHitFilter = cms.ESProducer("ClusterShapeHitFilterESProducer",
    ComponentName = cms.string('mixedTripletStepClusterShapeHitFilter'),
    PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/ITShapePhase2_all.par'),
    PixelShapeFileL1 = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/ITShapePhase2_all.par'),
    clusterChargeCut = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutTight')
    )
)
