import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase1PixelNewFPix_cff import phase1PixelNewFPix as _phase1PixelNewFPix

ClusterShapeHitFilterESProducer = cms.ESProducer("ClusterShapeHitFilterESProducer",
                                                        ComponentName = cms.string('ClusterShapeHitFilter'),
                                                        PixelShapeFile= cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par'),
                                                        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
                                                        )
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(ClusterShapeHitFilterESProducer,
    PixelShapeFile = 'RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1Tk.par'
)
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(ClusterShapeHitFilterESProducer,
    PixelShapeFile = 'RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1Tk.par'
)
from Configuration.Eras.Modifier_phase1PixelNewFPix_cff import phase1PixelNewFPix
_phase1PixelNewFPix.toModify(ClusterShapeHitFilterESProducer,
    PixelShapeFile = 'RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1TkNewFPix.par'
)
