import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlZMuMu AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlZMuMuFilterForSiPixelAli = copy.deepcopy(hltHighLevel)
ALCARECOTkAlZMuMuFilterForSiPixelAli.HLTPaths = ['pathALCARECOTkAlZMuMu']
ALCARECOTkAlZMuMuFilterForSiPixelAli.throw = True ## dont throw on unknown path names
ALCARECOTkAlZMuMuFilterForSiPixelAli.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")

from Alignment.CommonAlignmentProducer.ALCARECOPromptCalibProdSiPixelAli_cff import *
from Alignment.CommonAlignmentProducer.LSNumberFilter_cfi import *

# Ingredient: offlineBeamSpot
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import offlineBeamSpot

# Ingredient: AlignmentTrackSelector
# track selector for HighPurity tracks
#-- AlignmentTrackSelector
SiPixelAliHighPuritySelectorHGDimuon = SiPixelAliHighPuritySelector.clone(
    src = 'ALCARECOTkAlZMuMu'
)

# track selection for alignment
SiPixelAliTrackSelectorHGDimuon = SiPixelAliTrackSelector.clone(
	src = 'SiPixelAliTrackFitterHGDimuon'
)

# Ingredient: SiPixelAliTrackRefitter0
# refitting
SiPixelAliTrackRefitterHGDimuon0 = SiPixelAliTrackRefitter0.clone(
	src = 'SiPixelAliHighPuritySelectorHGDimuon'
)
SiPixelAliTrackRefitterHGDimuon1 = SiPixelAliTrackRefitterHGDimuon0.clone(
	src = 'SiPixelAliTrackSelectorHGDimuon'
)

#-- Alignment producer
SiPixelAliMilleAlignmentProducerHGDimuon = SiPixelAliMilleAlignmentProducer.clone(
    ParameterBuilder = dict(
      Selector = cms.PSet(
	alignParams = cms.vstring(
	  "TrackerP1PXBLadder,111111",
	  "TrackerP1PXECPanel,111111",
	)
      )
    ),
    tjTkAssociationMapTag = 'SiPixelAliTrackRefitterHGDimuon1',
    algoConfig = MillePedeAlignmentAlgorithm.clone(        
	binaryFile = 'milleBinaryHGDimuon_0.dat',
	treeFile = 'treeFileHGDimuon.root',
	monitorFile = 'millePedeMonitorHGDimuon.root',
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
SiPixelAliTrackerTrackHitFilterHGDimuon = SiPixelAliTrackerTrackHitFilter.clone(
	src = 'SiPixelAliTrackRefitterHGDimuon0'
)

# Ingredient: SiPixelAliSiPixelAliTrackFitter
SiPixelAliTrackFitterHGDimuon = SiPixelAliTrackFitter.clone(
	src = 'SiPixelAliTrackerTrackHitFilterHGDimuon'
)

SiPixelAliMillePedeFileConverterHGDimuon = cms.EDProducer("MillePedeFileConverter",
                                                         fileDir = cms.string(SiPixelAliMilleAlignmentProducerHGDimuon.algoConfig.fileDir.value()),
                                                         inputBinaryFile = cms.string(SiPixelAliMilleAlignmentProducerHGDimuon.algoConfig.binaryFile.value()),
                                                         fileBlobLabel = cms.string(''))

seqALCARECOPromptCalibProdSiPixelAliHGDiMu = cms.Sequence(ALCARECOTkAlZMuMuFilterForSiPixelAli*
                                                          lsNumberFilter*
                                                          offlineBeamSpot*
                                                          SiPixelAliHighPuritySelectorHGDimuon*
                                                          SiPixelAliTrackRefitterHGDimuon0*
                                                          SiPixelAliTrackerTrackHitFilterHGDimuon*
                                                          SiPixelAliTrackFitterHGDimuon*
                                                          SiPixelAliTrackSelectorHGDimuon*
                                                          SiPixelAliTrackRefitterHGDimuon1*
                                                          SiPixelAliMilleAlignmentProducerHGDimuon*
                                                          SiPixelAliMillePedeFileConverterHGDimuon)
