import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlMinBiasFilterForSiPixelAli = copy.deepcopy(hltHighLevel)
ALCARECOTkAlMinBiasFilterForSiPixelAli.HLTPaths = ['pathALCARECOTkAlMinBias']
ALCARECOTkAlMinBiasFilterForSiPixelAli.throw = True ## dont throw on unknown path names
ALCARECOTkAlMinBiasFilterForSiPixelAli.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")

# Adding geometry here, since this was also added to the process in the
# original alignment_BASE.py that was the configuration template used for mille
from Configuration.Geometry.GeometryIdeal_cff import *

# Ingredient: offlineBeamSpot
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import offlineBeamSpot

# Ingredient: AlignmentTrackSelector
# track selection for alignment
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import AlignmentTrackSelector
AlignmentTrackSelector.src = 'ALCARECOTkAlMinBias'  #'SiPixelAliTrackFitter' #'ALCARECOTkAlCosmicsCTF0T' #TkAlZMuMu' #MinBias' #'generalTracks' ## ALCARECOTkAlMuonIsolated # adjust to input file
AlignmentTrackSelector.pMin = 4.
AlignmentTrackSelector.ptMin = 0. #HIGHER CUT, LESS TRACKS, MORE EVENTS, LESS TIME THOUGH?????
AlignmentTrackSelector.ptMax = 200.
AlignmentTrackSelector.etaMin = -999.
AlignmentTrackSelector.etaMax = 999.
AlignmentTrackSelector.nHitMin = 10
AlignmentTrackSelector.nHitMin2D = 3
AlignmentTrackSelector.chi2nMax = 100.
AlignmentTrackSelector.applyMultiplicityFilter = False# True
AlignmentTrackSelector.maxMultiplicity = 1
AlignmentTrackSelector.minHitsPerSubDet.inPIXEL = 2


# Ingredient: SiPixelAliTrackRefitter0
# refitting
from RecoTracker.TrackProducer.TrackRefitters_cff import *
# In the following use
# TrackRefitter (normal tracks), TrackRefitterP5 (cosmics) or TrackRefitterBHM (beam halo)

SiPixelAliTrackRefitter0 = TrackRefitter.clone(
        src = 'AlignmentTrackSelector',   #'ALCARECOTkAlMinBias'#'ALCARECOTkAlCosmicsCTF0T' #'ALCARECOTkAlMuonIsolated'
        NavigationSchool = '',            # to avoid filling hit pattern
                                              )

# Alignment producer (which is a cms.Looper module and hence not added to the sequence)
from Alignment.CommonAlignmentProducer.AlignmentProducer_cff import *
#looper.parameterTypes = cms.vstring('Selector,RigidBody')
#looper.ParameterBuilder.parameterTypes = [
#    'SelectorRigid,RigidBody',
#    'SelectorBowed,BowedSurface'
#    ,'Selector2Bowed,TwoBowedSurfaces'
#    ]
looper.ParameterBuilder.Selector = cms.PSet(
    alignParams = cms.vstring(
        'TrackerTPBHalfBarrel,111111',
        'TrackerTPEHalfCylinder,111111',
#        'TrackerTPBLayer,111111',
#        'TrackerTPEHalfDisk,111111',
        'TrackerTIBHalfBarrel,ffffff', # or fff fff?
        'TrackerTOBHalfBarrel,ffffff', # dito...
        'TrackerTIDEndcap,ffffff',
        'TrackerTECEndcap,ffffff'
        )
    )

looper.doMisalignmentScenario = False #True

# If the above is true, you might want to choose the scenario:
#from Alignment.TrackerAlignment.Scenarios_cff import Tracker10pbScenario as Scenario # TrackerSurveyLASOnlyScenario

looper.MisalignmentScenario = cms.PSet(
    setRotations = cms.bool(True),
    setTranslations = cms.bool(True),
    seed = cms.int32(1234567),
    distribution = cms.string('fixed'), # gaussian, uniform (or so...)
    setError = cms.bool(True), #GF ???????
    TPBHalfBarrel1 = cms.PSet(
        dXlocal = cms.double(0.0020),
        dYlocal = cms.double(-0.0015),
        dZlocal = cms.double(0.0100),
        phiXlocal = cms.double(1.e-4),
        phiYlocal = cms.double(-2.e-4),
        phiZlocal = cms.double(5.e-4),
        ),
    TPBHalfBarrel2 = cms.PSet(
        dXlocal = cms.double(-0.0020),
        dYlocal = cms.double(0.0030),
        dZlocal = cms.double(-0.020),
        phiXlocal = cms.double(1.e-3),
        phiYlocal = cms.double(2.e-4),
        phiZlocal = cms.double(-2.e-4),
        ),
    TPEEndcap1 = cms.PSet(
        TPEHalfCylinder1 = cms.PSet(
            dXlocal = cms.double(0.0050),
            dYlocal = cms.double(0.0020),
            dZlocal = cms.double(-0.005),
            phiXlocal = cms.double(-1.e-5),
            phiYlocal = cms.double(2.e-3),
            phiZlocal = cms.double(2.e-5),
            ),
        TPEHalfCylinder2 = cms.PSet(
            dXlocal = cms.double(0.0020),
            dYlocal = cms.double(0.0030),
            dZlocal = cms.double(-0.01),
            phiXlocal = cms.double(1.e-4),
            phiYlocal = cms.double(-1.e-4),
            phiZlocal = cms.double(2.e-4),
            ),
        ),
    TPEEndcap2 = cms.PSet(
        TPEHalfCylinder1 = cms.PSet(
            dXlocal = cms.double(-0.0080),
            dYlocal = cms.double(0.0050),
            dZlocal = cms.double(-0.005),
            phiXlocal = cms.double(1.e-3),
            phiYlocal = cms.double(-3.e-4),
            phiZlocal = cms.double(2.e-4),
            ),
        TPEHalfCylinder2 = cms.PSet(
            dXlocal = cms.double(0.0020),
            dYlocal = cms.double(0.0030),
            dZlocal = cms.double(-0.005),
            phiXlocal = cms.double(-1.e-3),
            phiYlocal = cms.double(2.e-4),
            phiZlocal = cms.double(3.e-4),
            ),
        )
    )

looper.checkDbAlignmentValidity = False
looper.applyDbAlignment = True
looper.tjTkAssociationMapTag = 'SiPixelAliTrackFitter'

# assign by reference (i.e. could change MillePedeAlignmentAlgorithm as well):
looper.algoConfig = MillePedeAlignmentAlgorithm

#from Alignment.MillePedeAlignmentAlgorithm.PresigmaScenarios_cff import *
#looper.algoConfig.pedeSteerer.Presigmas.extend(TrackerShortTermPresigmas.Presigmas)
looper.algoConfig.mode = 'mille' #'mille' #'full' # 'pede' # 'full' # 'pedeSteerer'
#looper.algoConfig.mergeBinaryFiles = ['milleBinaryISN.dat']
#looper.algoConfig.mergeTreeFiles = ['treeFileISN_reg.root']
looper.algoConfig.binaryFile = 'milleBinaryISN.dat' # BVB: Remove this after it's take care of for the reading
looper.algoConfig.treeFile = 'treeFileISN.root' # BVB: Remove this

looper.algoConfig.TrajectoryFactory = BrokenLinesTrajectoryFactory
looper.algoConfig.TrajectoryFactory.MaterialEffects = 'BrokenLinesCoarse' #Coarse' #Fine' #'BreakPoints'
#looper.algoConfig.pedeSteerer.pedeCommand = '/afs/cern.ch/user/f/flucke/cms/pede/trunk_v69/pede_8GB'
looper.algoConfig.pedeSteerer.pedeCommand = '/afs/cern.ch/user/c/ckleinw/bin/rev125/pede'
#default is  sparseMINRES 6 0.8:                     <method>  n(iter)  Delta(F)
looper.algoConfig.pedeSteerer.method = 'inversion  5  0.8'  ##DNOONAN can be set to inversion instead (faster)
looper.algoConfig.pedeSteerer.options = cms.vstring(
    #'regularisation 1.0 0.05', # non-stated pre-sigma 50 mrad or 500 mum
    'entries 500',
    'chisqcut  30.0  4.5', #,
    'threads 1 1' #,
    #'outlierdownweighting 3','dwfractioncut 0.1'
    #'outlierdownweighting 5','dwfractioncut 0.2'
    )

looper.algoConfig.minNumHits = 8

looper.saveToDB = False

# Ingredient: SiPixelAliTrackerTrackHitFilter
import RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff as HitFilter
# Reference config at /afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/aliConfigTemplates/Cosmics38T_BL_default_cfg.py
SiPixelAliTrackerTrackHitFilter = HitFilter.TrackerTrackHitFilter.clone(
    src = 'SiPixelAliTrackRefitter0', #'ALCARECOTkAlCosmicsCTF0T',
    useTrajectories= False,#True, # for angle selections + pixel cluster charge
    minimumHits = 8,
    commands = [], # Ref. has equivalent pharse...
    detsToIgnore = [], #is default
    replaceWithInactiveHits = True, # needed for multiple scattering
    stripAllInvalidHits = False, #default
    rejectBadStoNHits = True,
    StoNcommands = ["ALL 18.0"], # 18 for tracker in peak mode, 5 for deconvolution mode
#    rejectLowAngleHits = True,
    TrackAngleCut = 0.1, # 0.35, # in rads, starting from the module surface; .35 for cosmcics ok, .17 for collision tracks
    usePixelQualityFlag = True
    )

# Ingredient: SiPixelAliSiPixelAliTrackFitter
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff as fitWithMaterial
SiPixelAliTrackFitter = fitWithMaterial.ctfWithMaterialTracks.clone(
        src = 'SiPixelAliTrackerTrackHitFilter',
        # TTRHBuilder = 'WithAngleAndTemplate', #should already be default
        NavigationSchool = ''
        )

# Ingredient: MillePedeFileConverter 
from Alignment.CommonAlignmentProducer.MillePedeFileConverter_cfi import millePedeFileConverter
# We configure the input file name of the millePedeFileConverter
#         with the output file name of the alignmentProducer (=looper).
# Like this we are sure that they are well connected.
SiPixelAliMillePedeFileConverter = millePedeFileConverter.clone(
        fileDir = looper.algoConfig.fileDir,
        binaryFile = looper.algoConfig.binaryFile,
        )

seqALCARECOPromptCalibProdSiPixelAli = cms.Sequence(ALCARECOTkAlMinBiasFilterForSiPixelAli*
                                                    offlineBeamSpot*
                                                    AlignmentTrackSelector*
                                                    SiPixelAliTrackRefitter0*
                                                    SiPixelAliTrackerTrackHitFilter*
                                                    SiPixelAliTrackFitter*
                                                    SiPixelAliMillePedeFileConverter)
