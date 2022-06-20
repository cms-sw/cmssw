import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
from Alignment.CommonAlignmentProducer.ALCARECOPromptCalibProdSiPixelAli_cff import *
ALCARECOTkAlMinBiasFilterForSiPixelAliHG = ALCARECOTkAlMinBiasFilterForSiPixelAli.clone()


from Alignment.CommonAlignmentProducer.LSNumberFilter_cfi import *

# Ingredient: offlineBeamSpot
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import offlineBeamSpot

# Ingredient: AlignmentTrackSelector
# track selector for HighPurity tracks
#-- AlignmentTrackSelector
SiPixelAliHighPuritySelectorHG = SiPixelAliHighPuritySelector.clone()

# track selection for alignment
SiPixelAliTrackSelectorHG = SiPixelAliTrackSelector.clone(
	src = 'SiPixelAliTrackFitterHG'
)

# Ingredient: SiPixelAliTrackRefitter0
# refitting
SiPixelAliTrackRefitterHG0 = SiPixelAliTrackRefitter0.clone(
	src = 'SiPixelAliHighPuritySelectorHG'
)
SiPixelAliTrackRefitterHG1 = SiPixelAliTrackRefitterHG0.clone(
	src = 'SiPixelAliTrackSelectorHG'
)

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
    tjTkAssociationMapTag = 'SiPixelAliTrackRefitterHG1',
    algoConfig = MillePedeAlignmentAlgorithm.clone(
	binaryFile = 'milleBinaryHG_0.dat',
	treeFile = 'treeFileHG.root',
	monitorFile = 'millePedeMonitorHG.root'
    )
)

# Ingredient: SiPixelAliTrackerTrackHitFilter
SiPixelAliTrackerTrackHitFilterHG = SiPixelAliTrackerTrackHitFilter.clone(
	src = 'SiPixelAliTrackRefitterHG0'
)

# Ingredient: SiPixelAliSiPixelAliTrackFitter
SiPixelAliTrackFitterHG = SiPixelAliTrackFitter.clone(
	src = 'SiPixelAliTrackerTrackHitFilterHG'
)

SiPixelAliMillePedeFileConverterHG = cms.EDProducer("MillePedeFileConverter",
                                                  fileDir = cms.string(SiPixelAliMilleAlignmentProducerHG.algoConfig.fileDir.value()),
                                                  inputBinaryFile = cms.string(SiPixelAliMilleAlignmentProducerHG.algoConfig.binaryFile.value()),
                                                  fileBlobLabel = cms.string(''),
                                                 )



seqALCARECOPromptCalibProdSiPixelAliHG = cms.Sequence(ALCARECOTkAlMinBiasFilterForSiPixelAliHG*
                                                    lsNumberFilter*
                                                    offlineBeamSpot*
                                                    SiPixelAliHighPuritySelectorHG*
                                                    SiPixelAliTrackRefitterHG0*
                                                    SiPixelAliTrackerTrackHitFilterHG*
                                                    SiPixelAliTrackFitterHG*
                                                    SiPixelAliTrackSelectorHG*
                                                    SiPixelAliTrackRefitterHG1*
                                                    SiPixelAliMilleAlignmentProducerHG*
                                                    SiPixelAliMillePedeFileConverterHG)
