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
onlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

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

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import stripCPEESProducer
hltESPStripCPEfromTrackAngle = stripCPEESProducer.clone(
  ComponentType = "StripCPEfromTrackAngle" ,
  ComponentName = "hltESPStripCPEfromTrackAngle",
  parameters = cms.PSet( 
    mLC_P2 = cms.double(0.3),
    mLC_P1 = cms.double(0.618),
    mLC_P0 = cms.double(-0.326),
#    useLegacyError = cms.bool( True ), # 50ns menu
#    maxChgOneMIP = cms.double( -6000.0 ), # 50ns menu
    useLegacyError = cms.bool(False) , # 25ns menu
    maxChgOneMIP = cms.double(6000.0) , #25ns menu
    mTEC_P1 = cms.double( 0.471 ),
    mTEC_P0 = cms.double( -1.885 ),
    mTOB_P0 = cms.double( -1.026 ),
    mTOB_P1 = cms.double( 0.253 ),
    mTIB_P0 = cms.double( -0.742 ),
    mTIB_P1 = cms.double( 0.202 ),
    mTID_P0 = cms.double( -1.427 ),
    mTID_P1 = cms.double( 0.433 )
  )
)

hltESPPixelCPEGeneric = cms.ESProducer( 
    "PixelCPEGenericESProducer",
    LoadTemplatesFromDB = cms.bool( True ),
    Alpha2Order = cms.bool( True ),
    ClusterProbComputationFlag = cms.int32( 0 ),
    useLAWidthFromDB = cms.bool( False ),
    lAOffset = cms.double( 0.0 ),
    lAWidthBPix = cms.double( 0.0 ),
    lAWidthFPix = cms.double( 0.0 ),
    doLorentzFromAlignment = cms.bool( False ),
    useLAFromDB = cms.bool( True ),
    xerr_barrel_l1 = cms.vdouble( 0.00115, 0.0012, 8.8E-4 ),
    yerr_barrel_l1 = cms.vdouble( 0.00375, 0.0023, 0.0025, 0.0025, 0.0023, 0.0023, 0.0021, 0.0021, 0.0024 ),
    xerr_barrel_ln = cms.vdouble( 0.00115, 0.0012, 8.8E-4 ),
    yerr_barrel_ln = cms.vdouble( 0.00375, 0.0023, 0.0025, 0.0025, 0.0023, 0.0023, 0.0021, 0.0021, 0.0024 ),
    xerr_endcap = cms.vdouble( 0.002, 0.002 ),
    yerr_endcap = cms.vdouble( 0.0021 ),
    xerr_barrel_l1_def = cms.double( 0.0103 ),
    yerr_barrel_l1_def = cms.double( 0.0021 ),
    xerr_barrel_ln_def = cms.double( 0.0103 ),
    yerr_barrel_ln_def = cms.double( 0.0021 ),
    xerr_endcap_def = cms.double( 0.002 ),
    yerr_endcap_def = cms.double( 7.5E-4 ),
    eff_charge_cut_highX = cms.double( 1.0 ),
    eff_charge_cut_highY = cms.double( 1.0 ),
    eff_charge_cut_lowX = cms.double( 0.0 ),
    eff_charge_cut_lowY = cms.double( 0.0 ),
    size_cutX = cms.double( 3.0 ),
    size_cutY = cms.double( 3.0 ),
    EdgeClusterErrorX = cms.double( 50.0 ),
    EdgeClusterErrorY = cms.double( 85.0 ),
    inflate_errors = cms.bool( False ),
    inflate_all_errors_no_trk_angle = cms.bool( False ),
    NoTemplateErrorsWhenNoTrkAngles = cms.bool( False ),
    UseErrorsFromTemplates = cms.bool( True ),
    TruncatePixelCharge = cms.bool( True ),
    IrradiationBiasCorrection = cms.bool( True ),
    DoCosmics = cms.bool( False ),
    isPhase2 = cms.bool( False ),
    SmallPitch = cms.bool( False ),
    ComponentName = cms.string( "hltESPPixelCPEGeneric" ),
    MagneticFieldRecord = cms.ESInputTag( "","" ),
    appendToDataLabel = cms.string( "" )
)

hltESPTTRHBWithTrackAngle = cms.ESProducer( 
    "TkTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" ),
    ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
    StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
    PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
    Matcher = cms.string( "StandardMatcher" ),
    Phase2StripCPE = cms.string( "" ),
    appendToDataLabel = cms.string( "" )
)

SiPixelAliTrackRefitterHLT0 = TrackRefitter.clone(
    src = 'SiPixelAliLooseSelectorHLT',   #'ALCARECOTkAlMinBias'#'ALCARECOTkAlCosmicsCTF0T' #'ALCARECOTkAlMuonIsolated'
    NavigationSchool = '',            # to avoid filling hit pattern
    TTRHBuilder = 'hltESPTTRHBWithTrackAngle'
)

SiPixelAliTrackRefitterHLT1 = SiPixelAliTrackRefitter0.clone(
	src = 'SiPixelAliTrackSelectorHLT'
)

#-- Alignment producer
from Alignment.MillePedeAlignmentAlgorithm.MillePedeAlignmentAlgorithm_cfi import *
from Alignment.CommonAlignmentProducer.AlignmentProducerAsAnalyzer_cff import AlignmentProducer
SiPixelAliMilleAlignmentProducerHLT = SiPixelAliMilleAlignmentProducer.clone(
    tjTkAssociationMapTag = 'SiPixelAliTrackRefitterHLT1',
    algoConfig = MillePedeAlignmentAlgorithm.clone(
        binaryFile = 'milleBinaryHLT_0.dat',
        treeFile = 'treeFileHLT.root',
        monitorFile = 'millePedeMonitorHLT.root'
    )
)
# Does anything else of the AlignmentProducer need to be overwritten ???




# Ingredient: SiPixelAliTrackerTrackHitFilterHLT
SiPixelAliTrackerTrackHitFilterHLT = SiPixelAliTrackerTrackHitFilter.clone(
	src = 'SiPixelAliTrackRefitterHLT0'
)


# Ingredient: SiPixelAliTrackFitterHLT
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff as fitWithMaterial
SiPixelAliTrackFitterHLT = fitWithMaterial.ctfWithMaterialTracks.clone(
    src = 'SiPixelAliTrackerTrackHitFilterHLT',
    # TTRHBuilder = 'hltESPTTRHBWithTrackAngle', #should already be default ???
    NavigationSchool = ''
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
