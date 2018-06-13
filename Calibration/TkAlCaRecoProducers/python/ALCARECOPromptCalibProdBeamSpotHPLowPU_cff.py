import FWCore.ParameterSet.Config as cms


from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdBeamSpotHP_cff import *

alcaBeamSpotProducerHPLowPU = alcaBeamSpotProducerHP.clone()
alcaBeamSpotProducerHPLowPU.PVFitter.minVertexNTracks = cms.untracked.uint32(0)
alcaBeamSpotProducerHPLowPU.PVFitter.useOnlyFirstPV = cms.untracked.bool(False)
alcaBeamSpotProducerHPLowPU.PVFitter.minSumPt = cms.untracked.double(0.)
alcaBeamSpotProducerHPLowPU.PVFitter.errorScale = cms.untracked.double(0.9)


seqALCARECOPromptCalibProdBeamSpotHPLowPU = cms.Sequence(ALCARECOTkAlMinBiasFilterForBSHP *
                                                         ALCARECOHltFilterForBSHP *
                                                         alcaBeamSpotProducerHPLowPU)
