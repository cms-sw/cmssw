import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlMinBiasFilterForSiPixelAliHLTHG = hltHighLevel.clone(
    HLTPaths = ['pathALCARECOTkAlHLTTracks'],
    throw = True, ## dont throw on unknown path names,
    TriggerResultsTag = "TriggerResults::RECO"
)

from Alignment.CommonAlignmentProducer.ALCARECOPromptCalibProdSiPixelAliHLT_cff import *
from Alignment.CommonAlignmentProducer.LSNumberFilter_cfi import *

# Ingredient: AlignmentTrackSelector
# track selector for HighPurity tracks
#-- AlignmentTrackSelector
SiPixelAliLooseSelectorHLTHG = SiPixelAliLooseSelectorHLT.clone(
    src = 'ALCARECOTkAlHLTTracks',
)

# track selection for alignment
SiPixelAliTrackSelectorHLTHG = SiPixelAliTrackSelectorHLT.clone(
    src = 'SiPixelAliTrackFitterHLTHG'
)

# Ingredient: SiPixelAliTrackRefitter0
# refitting
SiPixelAliTrackRefitterHLTHG0 = SiPixelAliTrackRefitterHLT0.clone(
    src = 'SiPixelAliLooseSelectorHLTHG'
)
SiPixelAliTrackRefitterHLTHG1 = SiPixelAliTrackRefitterHLTHG0.clone(
    src = 'SiPixelAliTrackSelectorHLTHG'
)

#-- Alignment producer
SiPixelAliMilleAlignmentProducerHLTHG = SiPixelAliMilleAlignmentProducerHLT.clone(
    ParameterBuilder = dict(
        Selector = cms.PSet(
            alignParams = cms.vstring(
                "TrackerP1PXBLadder,111111",
                "TrackerP1PXECPanel,111111",
            )
        )
    ),
    tjTkAssociationMapTag = 'SiPixelAliTrackRefitterHLTHG1',
    algoConfig = MillePedeAlignmentAlgorithm.clone(
        binaryFile = 'milleBinaryHLTHG_0.dat',
        treeFile = 'treeFileHLTHG.root',
        monitorFile = 'millePedeMonitorHLTHG.root'
    )
)

# Ingredient: SiPixelAliTrackerTrackHitFilter
SiPixelAliTrackerTrackHitFilterHLTHG = SiPixelAliTrackerTrackHitFilterHLT.clone(
    src = 'SiPixelAliTrackRefitterHLTHG0',
    usePixelQualityFlag = False
)

# Ingredient: SiPixelAliSiPixelAliTrackFitter
SiPixelAliTrackFitterHLTHG = SiPixelAliTrackFitterHLT.clone(
    src = 'SiPixelAliTrackerTrackHitFilterHLTHG'
)

SiPixelAliMillePedeFileConverterHLTHG = cms.EDProducer(
    "MillePedeFileConverter",
    fileDir = cms.string(SiPixelAliMilleAlignmentProducerHLTHG.algoConfig.fileDir.value()),
    inputBinaryFile = cms.string(SiPixelAliMilleAlignmentProducerHLTHG.algoConfig.binaryFile.value()),
    fileBlobLabel = cms.string('')
)

seqALCARECOPromptCalibProdSiPixelAliHLTHG = cms.Sequence(
    ALCARECOTkAlMinBiasFilterForSiPixelAliHLTHG*
    LSNumberFilter*
    onlineBeamSpot*
    SiPixelAliLooseSelectorHLTHG*
    SiPixelAliTrackRefitterHLTHG0*
    SiPixelAliTrackerTrackHitFilterHLTHG*
    SiPixelAliTrackFitterHLTHG*
    SiPixelAliTrackSelectorHLTHG*
    SiPixelAliTrackRefitterHLTHG1*
    SiPixelAliMilleAlignmentProducerHLTHG*
    SiPixelAliMillePedeFileConverterHLTHG
)
