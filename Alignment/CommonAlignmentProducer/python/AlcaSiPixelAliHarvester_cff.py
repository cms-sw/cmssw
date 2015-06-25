import FWCore.ParameterSet.Config as cms
import copy

from Alignment.MillePedeAlignmentAlgorithm.MillePedeAlignmentAlgorithm_cfi import *
from Alignment.CommonAlignmentProducer.TrackerAlignmentProducerForPCL_cff import AlignmentProducer
SiPixelAliPedeAlignmentProducer = copy.deepcopy(AlignmentProducer)

SiPixelAliPedeAlignmentProducer.ParameterBuilder.Selector = cms.PSet(
    alignParams = cms.vstring(
        'TrackerTPBHalfBarrel,111111',
        'TrackerTPEHalfCylinder,111111',

        'TrackerTIBHalfBarrel,ffffff',
        'TrackerTOBHalfBarrel,rrrrrr',
        'TrackerTIDEndcap,ffffff',
        'TrackerTECEndcap,ffffff'
        )
    )

SiPixelAliPedeAlignmentProducer.doMisalignmentScenario = False #True

SiPixelAliPedeAlignmentProducer.MisalignmentScenario = cms.PSet(
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

SiPixelAliPedeAlignmentProducer.checkDbAlignmentValidity = False
SiPixelAliPedeAlignmentProducer.applyDbAlignment = True
SiPixelAliPedeAlignmentProducer.tjTkAssociationMapTag = 'TrackRefitter2'

SiPixelAliPedeAlignmentProducer.algoConfig = MillePedeAlignmentAlgorithm
SiPixelAliPedeAlignmentProducer.algoConfig.mode = 'pede'
# FIXME: this needs to be addressed
SiPixelAliPedeAlignmentProducer.algoConfig.mergeBinaryFiles = [
				'milleBinary0.dat'
				]
SiPixelAliPedeAlignmentProducer.algoConfig.monitorFile = 'millePedeMonitor_pede.root'
SiPixelAliPedeAlignmentProducer.algoConfig.binaryFile = ''
SiPixelAliPedeAlignmentProducer.algoConfig.TrajectoryFactory = BrokenLinesBzeroTrajectoryFactory
SiPixelAliPedeAlignmentProducer.algoConfig.TrajectoryFactory.MomentumEstimate = 10
SiPixelAliPedeAlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = 'BrokenLinesCoarse' #Coarse' #Fine' #'BreakPoints'
SiPixelAliPedeAlignmentProducer.algoConfig.TrajectoryFactory.UseInvalidHits = True # to account for multiple scattering in these layers
SiPixelAliPedeAlignmentProducer.algoConfig.pedeSteerer.pedeCommand = 'pede'
SiPixelAliPedeAlignmentProducer.algoConfig.pedeSteerer.method = 'inversion  5  0.8' 
SiPixelAliPedeAlignmentProducer.algoConfig.pedeSteerer.options = cms.vstring(
    #'regularisation 1.0 0.05', # non-stated pre-sigma 50 mrad or 500 mum
    'entries 500',
    'chisqcut  30.0  4.5', #,
    'threads 1 1' #,
    #'outlierdownweighting 3','dwfractioncut 0.1'
    #'outlierdownweighting 5','dwfractioncut 0.2'
    )
SiPixelAliPedeAlignmentProducer.algoConfig.minNumHits = 8


from Alignment.FileConverterPlaceHolder.fileconverterplaceholder_cfi import *

ALCAHARVESTSiPixelAli = cms.Sequence(SiPixelAliPedeAlignmentProducer*
                                     demo)
