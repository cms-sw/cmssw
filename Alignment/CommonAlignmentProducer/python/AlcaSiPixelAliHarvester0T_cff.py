import FWCore.ParameterSet.Config as cms
import copy

SiPixelAliMilleFileExtractor = cms.EDAnalyzer("MillePedeFileExtractor",
    fileBlobInputTag = cms.InputTag("SiPixelAliMillePedeFileConverter",''),
    # File names the Extractor will use to write the fileblobs in the root
    # file as real binary files to disk, so that the pede step can read them.
    # This includes the formatting directive "%04d" which will be expanded to
    # 0000, 0001, 0002,...
    outputBinaryFile = cms.string('pedeBinary%04d.dat'))

from Alignment.MillePedeAlignmentAlgorithm.MillePedeAlignmentAlgorithm_cfi import *
from Alignment.CommonAlignmentProducer.AlignmentProducerAsAnalyzer_cff import AlignmentProducer
SiPixelAliPedeAlignmentProducer = copy.deepcopy(AlignmentProducer)

SiPixelAliPedeAlignmentProducer.ParameterBuilder.Selector = cms.PSet(
    alignParams = cms.vstring(
        "PixelHalfBarrels,111111",
        "PXECHalfCylinders,111111",
        )
    )

SiPixelAliPedeAlignmentProducer.doMisalignmentScenario = False #True


SiPixelAliPedeAlignmentProducer.checkDbAlignmentValidity = False
SiPixelAliPedeAlignmentProducer.applyDbAlignment = True
SiPixelAliPedeAlignmentProducer.tjTkAssociationMapTag = 'TrackRefitter2'

SiPixelAliPedeAlignmentProducer.algoConfig = MillePedeAlignmentAlgorithm
SiPixelAliPedeAlignmentProducer.algoConfig.mode = 'pede'
SiPixelAliPedeAlignmentProducer.algoConfig.runAtPCL = True
SiPixelAliPedeAlignmentProducer.algoConfig.mergeBinaryFiles = [SiPixelAliMilleFileExtractor.outputBinaryFile.value()]
SiPixelAliPedeAlignmentProducer.algoConfig.binaryFile = ''
SiPixelAliPedeAlignmentProducer.algoConfig.TrajectoryFactory = cms.PSet(
      BrokenLinesBzeroTrajectoryFactory # For 0T collisions
      )
SiPixelAliPedeAlignmentProducer.algoConfig.TrajectoryFactory.MomentumEstimate = 5 #for 0T Collisions      
      
SiPixelAliPedeAlignmentProducer.algoConfig.pedeSteerer.pedeCommand = 'pede'
SiPixelAliPedeAlignmentProducer.algoConfig.pedeSteerer.method = 'inversion  5  0.8'
SiPixelAliPedeAlignmentProducer.algoConfig.pedeSteerer.options = cms.vstring(
    #'regularisation 1.0 0.05', # non-stated pre-sigma 50 mrad or 500 mum
     'entries 500',
     'chisqcut  30.0  4.5',
     'threads 1 1',
     'closeandreopen'
     #'outlierdownweighting 3','dwfractioncut 0.1'
     #'outlierdownweighting 5','dwfractioncut 0.2'
    )
SiPixelAliPedeAlignmentProducer.algoConfig.minNumHits = 10
SiPixelAliPedeAlignmentProducer.saveToDB = True



ALCAHARVESTSiPixelAli = cms.Sequence(SiPixelAliMilleFileExtractor*
                                     SiPixelAliPedeAlignmentProducer)
