import FWCore.ParameterSet.Config as cms
# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
from Alignment.CommonAlignmentProducer.ALCARECOPromptCalibProdSiPixelAli_cff import *
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlMinBiasFilterForSiPixelAliHLT = hltHighLevel.clone(
    HLTPaths = ['pathALCARECOTkAlHLTTracks'], # ???
    throw = True, ## dont throw on unknown path names,
    TriggerResultsTag = "TriggerResults::RECO"
)

from Alignment.CommonAlignmentProducer.LSNumberFilter_cfi import *

# Ingredient: onlineBeamSpot
import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
onlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone(
    useBSOnlineRecords = True,
    timeThreshold = 999999
)

# Ingredient: ALCARECOTkAlMinBiasHLT
from  Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_cff import ALCARECOTkAlMinBias
ALCARECOTkAlMinBiasHLTTracks = ALCARECOTkAlMinBias.clone(
    src = cms.InputTag("hltMergedTracks")
)

# Ingredient: AlignmentTrackSelector
# track selector for HighPurity tracks
#-- AlignmentTrackSelector
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import AlignmentTrackSelector
SiPixelAliLooseSelectorHLT = AlignmentTrackSelector.clone(
    applyBasicCuts = True,
    #filter = True,
    src = 'ALCARECOTkAlMinBiasHLTTracks',
    trackQualities = ["loose"],
    pMin = 4.,
)

# track selection for alignment
SiPixelAliTrackSelectorHLT = SiPixelAliTrackSelector.clone( 
	src = 'SiPixelAliTrackFitterHLT',
)

# Ingredient: SiPixelAliTrackRefitter0
# refitting
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoTracker.TrackProducer.TrackRefitter_cfi import *
# In the following use
# TrackRefitter (normal tracks), TrackRefitterP5 (cosmics) or TrackRefitterBHM (beam halo)

# Ingredient: import HLT CPE ESProducers directly to avoid going out of synch
from Alignment.CommonAlignmentProducer.HLTModulesIncludes_cff import hltESPStripCPEfromTrackAngle, hltESPPixelCPEGeneric, hltESPTTRHBWithTrackAngle

SiPixelAliTrackRefitterHLT0 = TrackRefitter.clone(
    src = 'SiPixelAliLooseSelectorHLT',   #'ALCARECOTkAlMinBias'#'ALCARECOTkAlCosmicsCTF0T' #'ALCARECOTkAlMuonIsolated'
    NavigationSchool = '',            # to avoid filling hit pattern
    TTRHBuilder = 'hltESPTTRHBWithTrackAngle',
    beamSpot = 'onlineBeamSpot'
)

SiPixelAliTrackRefitterHLT1 = SiPixelAliTrackRefitter0.clone(
    src = 'SiPixelAliTrackSelectorHLT',
    TTRHBuilder = 'hltESPTTRHBWithTrackAngle',
    beamSpot = 'onlineBeamSpot'
)

#-- Alignment producer
from Alignment.MillePedeAlignmentAlgorithm.MillePedeAlignmentAlgorithm_cfi import *
from Alignment.CommonAlignmentProducer.AlignmentProducerAsAnalyzer_cff import AlignmentProducer
SiPixelAliMilleAlignmentProducerHLT = SiPixelAliMilleAlignmentProducer.clone(
    beamSpotTag = 'onlineBeamSpot',
    tjTkAssociationMapTag = 'SiPixelAliTrackRefitterHLT1',
    algoConfig = MillePedeAlignmentAlgorithm.clone(
        binaryFile = 'milleBinaryHLT_0.dat',
        treeFile = 'treeFileHLT.root',
        monitorFile = 'millePedeMonitorHLT.root'
    )
)

# Ingredient: SiPixelAliTrackerTrackHitFilterHLT
SiPixelAliTrackerTrackHitFilterHLT = SiPixelAliTrackerTrackHitFilter.clone(
	src = 'SiPixelAliTrackRefitterHLT0'
)

# Ingredient: SiPixelAliTrackFitterHLT
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff as fitWithMaterial
SiPixelAliTrackFitterHLT = fitWithMaterial.ctfWithMaterialTracks.clone(
    src = 'SiPixelAliTrackerTrackHitFilterHLT',
    NavigationSchool = '',
    TTRHBuilder = 'hltESPTTRHBWithTrackAngle',
    beamSpot = 'onlineBeamSpot'
)

SiPixelAliMillePedeFileConverterHLT = cms.EDProducer( 
    "MillePedeFileConverter",
    fileDir = cms.string(SiPixelAliMilleAlignmentProducerHLT.algoConfig.fileDir.value()),
    inputBinaryFile = cms.string(SiPixelAliMilleAlignmentProducerHLT.algoConfig.binaryFile.value()),
    fileBlobLabel = cms.string(''),
)

seqALCARECOPromptCalibProdSiPixelAliHLT = cms.Sequence(
    ALCARECOTkAlMinBiasFilterForSiPixelAliHLT*
    LSNumberFilter*
    onlineBeamSpot*
    SiPixelAliLooseSelectorHLT*
    SiPixelAliTrackRefitterHLT0*
    SiPixelAliTrackerTrackHitFilterHLT*
    SiPixelAliTrackFitterHLT*
    SiPixelAliTrackSelectorHLT*
    SiPixelAliTrackRefitterHLT1*
    SiPixelAliMilleAlignmentProducerHLT*
    SiPixelAliMillePedeFileConverterHLT
)
