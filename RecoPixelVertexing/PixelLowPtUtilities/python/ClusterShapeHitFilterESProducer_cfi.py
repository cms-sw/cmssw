import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelLowPtUtilities.clusterShapeHitFilterESProducer_cfi import clusterShapeHitFilterESProducer
ClusterShapeHitFilterESProducer = clusterShapeHitFilterESProducer.clone(ComponentName = 'ClusterShapeHitFilter',
                                                                        PixelShapeFile = 'RecoPixelVertexing/PixelLowPtUtilities/data/pixelShapePhase0.par',
                                                                        PixelShapeFileL1 = 'RecoPixelVertexing/PixelLowPtUtilities/data/pixelShapePhase0.par',
                                                                        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
                                                                        isPhase2 = False)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(ClusterShapeHitFilterESProducer,
    PixelShapeFile = 'RecoPixelVertexing/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par',
    PixelShapeFileL1 = 'RecoPixelVertexing/PixelLowPtUtilities/data/pixelShapePhase1_loose.par',
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(ClusterShapeHitFilterESProducer,
    isPhase2 = True,
    PixelShapeFile = 'RecoPixelVertexing/PixelLowPtUtilities/data/ITShapePhase2_all.par',
    PixelShapeFileL1 = 'RecoPixelVertexing/PixelLowPtUtilities/data/ITShapePhase2_all.par',
)
