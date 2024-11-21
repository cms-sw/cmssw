import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlZMuMu AlcaReco
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlZMuMuFilterForSiPixelAliHLT = hltHighLevel.clone(
    HLTPaths = ['pathALCARECOTkAlHLTTracksZMuMu'],
    throw = True, ## dont throw on unknown path names,
    TriggerResultsTag = "TriggerResults::RECO"
)

from Alignment.CommonAlignmentProducer.ALCARECOPromptCalibProdSiPixelAliHLT_cff import *
from Alignment.CommonAlignmentProducer.LSNumberFilter_cfi import *

# Ingredient: offlineBeamSpot
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import offlineBeamSpot

# Ingredient: AlignmentTrackSelector
# track selector for HighPurity tracks
#-- AlignmentTrackSelector
SiPixelAliLooseSelectorHLTHGDimuon = SiPixelAliLooseSelectorHLT.clone(
    src = 'ALCARECOTkAlHLTTracksZMuMu',
    etaMax = 3.0,
    etaMin = -3.0,
    filter = True,
    pMin = 8.0,
)

# track selection for alignment
SiPixelAliTrackSelectorHLTHGDimuon = SiPixelAliTrackSelectorHLT.clone(
    src = 'SiPixelAliTrackFitterHLTHGDimuon',
    applyMultiplicityFilter = True,
    d0Max = 50.0,
    d0Min = -50.0,
    etaMax = 3.0,
    etaMin = -3.0,
    filter = True,
    maxMultiplicity = 2,
    minHitChargeStrip = 20.0,
    minHitIsolation = 0.01,
    minMultiplicity = 2,
    nHighestPt = 2,
    nHitMin = 10,
    pMin = 3.0,
    ptMin = 15.0,
    TwoBodyDecaySelector = dict(applyChargeFilter = True,
                                applyMassrangeFilter = True,
                                maxXMass = 95.8,
                                minXMass = 85.8),
    minHitsPerSubDet = dict(inPIXEL = 1)
)

# Ingredient: SiPixelAliTrackRefitter0
# refitting
SiPixelAliTrackRefitterHLTHGDimuon0 = SiPixelAliTrackRefitterHLT0.clone(
	src = 'SiPixelAliLooseSelectorHLTHGDimuon'
)
SiPixelAliTrackRefitterHLTHGDimuon1 = SiPixelAliTrackRefitterHLTHGDimuon0.clone(
	src = 'SiPixelAliTrackSelectorHLTHGDimuon'
)

#-- Alignment producer
SiPixelAliMilleAlignmentProducerHLTHGDimuon = SiPixelAliMilleAlignmentProducerHLT.clone(
    ParameterBuilder = dict(
      Selector = cms.PSet(
	alignParams = cms.vstring(
	  "TrackerP1PXBLadder,111111",
	  "TrackerP1PXECPanel,111111",
	)
      )
    ),
    tjTkAssociationMapTag = 'SiPixelAliTrackRefitterHLTHGDimuon1',
    algoConfig = MillePedeAlignmentAlgorithm.clone(        
	binaryFile = 'milleBinaryHLTHGDimuon_0.dat',
	treeFile = 'treeFileHLTHGDimuon.root',
	monitorFile = 'millePedeMonitorHLTHGDimuon.root',
        minNumHits = 8,
        skipGlobalPositionRcdCheck = True,
        TrajectoryFactory = cms.PSet(
            AllowZeroMaterial = cms.bool(False),
            Chi2Cut = cms.double(10000.0),
            ConstructTsosWithErrors = cms.bool(False),
            EstimatorParameters = cms.PSet(
                MaxIterationDifference = cms.untracked.double(0.01),
                MaxIterations = cms.untracked.int32(100),
                RobustificationConstant = cms.untracked.double(1.0),
                UseInvariantMass = cms.untracked.bool(True)
            ),
            IncludeAPEs = cms.bool(False),
            MaterialEffects = cms.string('LocalGBL'),
            NSigmaCut = cms.double(100.0),
            ParticleProperties = cms.PSet(
                PrimaryMass = cms.double(91.1061),
                PrimaryWidth = cms.double(1.7678),
                SecondaryMass = cms.double(0.105658)
            ),
            PropagationDirection = cms.string('alongMomentum'),
            TrajectoryFactoryName = cms.string('TwoBodyDecayTrajectoryFactory'),
            UseBeamSpot = cms.bool(False),
            UseHitWithoutDet = cms.bool(True),
            UseInvalidHits = cms.bool(True),
            UseProjectedHits = cms.bool(True),
            UseRefittedState = cms.bool(True)
        )
    )
)

# Ingredient: SiPixelAliTrackerTrackHitFilter
SiPixelAliTrackerTrackHitFilterHLTHGDimuon = SiPixelAliTrackerTrackHitFilterHLT.clone(
    src = 'SiPixelAliTrackRefitterHLTHGDimuon0',
    TrackAngleCut = 0.087,
    minimumHits = 10,
    usePixelQualityFlag = False
)

# Ingredient: SiPixelAliSiPixelAliTrackFitter
SiPixelAliTrackFitterHLTHGDimuon = SiPixelAliTrackFitterHLT.clone(
    src = 'SiPixelAliTrackerTrackHitFilterHLTHGDimuon'
)

SiPixelAliMillePedeFileConverterHLTHGDimuon = cms.EDProducer(
    "MillePedeFileConverter",
    fileDir = cms.string(SiPixelAliMilleAlignmentProducerHLTHGDimuon.algoConfig.fileDir.value()),
    inputBinaryFile = cms.string(SiPixelAliMilleAlignmentProducerHLTHGDimuon.algoConfig.binaryFile.value()),
    fileBlobLabel = cms.string('')
)

seqALCARECOPromptCalibProdSiPixelAliHLTHGDiMu = cms.Sequence(
    ALCARECOTkAlZMuMuFilterForSiPixelAliHLT*
    LSNumberFilter*
    offlineBeamSpot*
    SiPixelAliLooseSelectorHLTHGDimuon*
    SiPixelAliTrackRefitterHLTHGDimuon0*
    SiPixelAliTrackerTrackHitFilterHLTHGDimuon*
    SiPixelAliTrackFitterHLTHGDimuon*
    SiPixelAliTrackSelectorHLTHGDimuon*
    SiPixelAliTrackRefitterHLTHGDimuon1*
    SiPixelAliMilleAlignmentProducerHLTHGDimuon*
    SiPixelAliMillePedeFileConverterHLTHGDimuon
)
