import FWCore.ParameterSet.Config as cms




# Alignment producer
from Alignment.CommonAlignmentProducer.AlignmentProducer_cff import *

#process.AlignmentProducer.parameterTypes = cms.vstring('Selector,RigidBody')
#process.AlignmentProducer.ParameterBuilder.parameterTypes = [
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

#FIXME: is this needed given the above parameters?
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


looper.tjTkAssociationMapTag = 'TrackFitter'

# assign by reference (i.e. could change MillePedeAlignmentAlgorithm as well):
looper.algoConfig = MillePedeAlignmentAlgorithm

#from Alignment.MillePedeAlignmentAlgorithm.PresigmaScenarios_cff import *
#looper.algoConfig.pedeSteerer.Presigmas.extend(TrackerShortTermPresigmas.Presigmas)
looper.algoConfig.mode = 'pede' #'mille' #'full' # 'pede' # 'full' # 'pedeSteerer'
#looper.algoConfig.mergeBinaryFiles = ['milleBinaryISN.dat']
#looper.algoConfig.mergeTreeFiles = ['treeFileISN_reg.root']
looper.algoConfig.binaryFile = 'milleBinaryISN.dat'
looper.algoConfig.treeFile = 'treeFileISN.root'




looper.algoConfig.TrajectoryFactory = BrokenLinesTrajectoryFactory
looper.algoConfig.TrajectoryFactory.MaterialEffects = 'BrokenLinesCoarse' #Coarse' #Fine' #'BreakPoints'
#looper.algoConfig.pedeSteerer.pedeCommand = '/afs/cern.ch/user/f/flucke/cms/pede/trunk_v69/pede_8GB'
#FIXME: this needs to come from the release
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

from Alignment.FileConverterPlaceHolder.fileconverterplaceholder_cfi import *

ALCAHARVESTSiPixelAli = cms.Sequence(demo)
