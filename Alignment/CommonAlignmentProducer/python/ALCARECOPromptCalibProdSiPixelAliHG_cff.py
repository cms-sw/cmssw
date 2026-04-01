import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
from Alignment.CommonAlignmentProducer.ALCARECOPromptCalibProdSiPixelAli_cff import *

#-- Alignment producer
SiPixelAliMilleAlignmentProducerHG = SiPixelAliMilleAlignmentProducer.clone(
    ParameterBuilder = dict(
      Selector = cms.PSet(
	alignParams = cms.vstring(
	  "TrackerP1PXBLadder,111111",
	  "TrackerP1PXECPanel,111111",
	)
      )
    ),
    tjTkAssociationMapTag = 'SiPixelAliTrackRefitter1',
    algoConfig = MillePedeAlignmentAlgorithm.clone(
	binaryFile = 'milleBinaryHG_0.dat',
	treeFile = 'treeFileHG.root',
	monitorFile = 'millePedeMonitorHG.root'
    )
)

SiPixelAliMillePedeFileConverterHG = cms.EDProducer("MillePedeFileConverter",
                                                    fileDir = cms.string(SiPixelAliMilleAlignmentProducerHG.algoConfig.fileDir.value()),
                                                    inputBinaryFile = cms.string(SiPixelAliMilleAlignmentProducerHG.algoConfig.binaryFile.value()),
                                                    fileBlobLabel = cms.string(''))

seqALCARECOPromptCalibProdSiPixelAliHG = cms.Sequence(ALCARECOTkAlMinBiasFilterForSiPixelAli*
                                                      LSNumberFilter*
                                                      offlineBeamSpot*
                                                      SiPixelAliHighPuritySelector*
                                                      SiPixelAliTrackRefitter0*
                                                      SiPixelAliTrackerTrackHitFilter*
                                                      SiPixelAliTrackFitter*
                                                      SiPixelAliTrackSelector*
                                                      SiPixelAliTrackRefitter1*
                                                      SiPixelAliMilleAlignmentProducerHG*
                                                      SiPixelAliMillePedeFileConverterHG)
