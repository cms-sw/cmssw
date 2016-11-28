import FWCore.ParameterSet.Config as cms

   
def customizeHLTPhaseIPixelGeom(process):

	process.ClusterShapeHitFilterESProducer.PixelShapeFile = 'RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1TkNewFPix.par'
	process.hltSiPixelDigis.UsePhase1 = cms.bool( True )
	process.hltSiPixelDigisRegForBTag.UsePhase1 = cms.bool( True )
	process.hltSiPixelDigisReg.UsePhase1 = cms.bool( True )


	return process
