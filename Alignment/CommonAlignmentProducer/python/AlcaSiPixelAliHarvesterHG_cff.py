import FWCore.ParameterSet.Config as cms
import copy

SiPixelAliMilleFileExtractorHG = cms.EDAnalyzer("MillePedeFileExtractor",
    fileBlobInputTag = cms.InputTag("SiPixelAliMillePedeFileConverterHG",''),
    fileDir = cms.string('HGalignment/'),
    # File names the Extractor will use to write the fileblobs in the root
    # file as real binary files to disk, so that the pede step can read them.
    # This includes the formatting directive "%04d" which will be expanded to
    # 0000, 0001, 0002,...
    outputBinaryFile = cms.string('pedeBinaryHG%04d.dat'))

from Alignment.MillePedeAlignmentAlgorithm.MillePedeAlignmentAlgorithm_cfi import *
from Alignment.CommonAlignmentProducer.AlignmentProducerAsAnalyzer_cff import AlignmentProducer
SiPixelAliPedeAlignmentProducerHG = copy.deepcopy(AlignmentProducer)

from Alignment.MillePedeAlignmentAlgorithm.MillePedeDQMModule_cff import *


SiPixelAliPedeAlignmentProducerHG.ParameterBuilder.Selector = cms.PSet(
    alignParams = cms.vstring(
        "TrackerP1PXBLadder,111111",
        "TrackerP1PXECPanel,111111",
        )
    )

SiPixelAliPedeAlignmentProducerHG.doMisalignmentScenario = False #True

SiPixelAliPedeAlignmentProducerHG.checkDbAlignmentValidity = False
SiPixelAliPedeAlignmentProducerHG.applyDbAlignment = True
SiPixelAliPedeAlignmentProducerHG.tjTkAssociationMapTag = 'TrackRefitter2'

SiPixelAliPedeAlignmentProducerHG.algoConfig = MillePedeAlignmentAlgorithm.clone()
SiPixelAliPedeAlignmentProducerHG.algoConfig.mode = 'pede'
SiPixelAliPedeAlignmentProducerHG.algoConfig.runAtPCL = True
SiPixelAliPedeAlignmentProducerHG.algoConfig.mergeBinaryFiles = [SiPixelAliMilleFileExtractorHG.outputBinaryFile.value()]
SiPixelAliPedeAlignmentProducerHG.algoConfig.binaryFile = ''
SiPixelAliPedeAlignmentProducerHG.algoConfig.TrajectoryFactory = cms.PSet(
      #process.BrokenLinesBzeroTrajectoryFactory
      BrokenLinesTrajectoryFactory
      )
SiPixelAliPedeAlignmentProducerHG.algoConfig.pedeSteerer.pedeCommand = 'pede'
SiPixelAliPedeAlignmentProducerHG.algoConfig.pedeSteerer.method = 'inversion  5  0.8'
SiPixelAliPedeAlignmentProducerHG.algoConfig.pedeSteerer.options = cms.vstring(
    #'regularisation 1.0 0.05', # non-stated pre-sigma 50 mrad or 500 mum
     #  ~'entries 500',
     'entries 10',
     'chisqcut  30.0  4.5',
     'threads 1 1',
     'closeandreopen',
     'skipemptycons' 
     #'outlierdownweighting 3','dwfractioncut 0.1'
     #'outlierdownweighting 5','dwfractioncut 0.2'
    )
#  ~SiPixelAliPedeAlignmentProducerHG.algoConfig.minNumHits = 10
SiPixelAliPedeAlignmentProducerHG.algoConfig.minNumHits = 0
SiPixelAliPedeAlignmentProducerHG.saveToDB = True

SiPixelAliPedeAlignmentProducerHG.algoConfig.fileDir = 'HGalignment/'
SiPixelAliPedeAlignmentProducerHG.algoConfig.pedeSteerer.fileDir = 'HGalignment/'
SiPixelAliPedeAlignmentProducerHG.algoConfig.pedeSteerer.runDir = cms.untracked.string('HGalignment/')
SiPixelAliPedeAlignmentProducerHG.algoConfig.pedeReader.fileDir = 'HGalignment/'
SiPixelAliPedeAlignmentProducerHG.algoConfig.MillePedeFileReader.fileDir = "HGalignment/"
SiPixelAliPedeAlignmentProducerHG.algoConfig.MillePedeFileReader.isHG = True

SiPixelAliPedeAlignmentProducerHG.algoConfig.pedeSteerer.steerFile = 'pedeSteerHG'
SiPixelAliPedeAlignmentProducerHG.algoConfig.pedeSteerer.pedeDump = 'pedeHG.dump'

SiPixelAliDQMModuleHG = SiPixelAliDQMModule.clone()
SiPixelAliDQMModuleHG.MillePedeFileReader.fileDir = "HGalignment/"
SiPixelAliDQMModuleHG.MillePedeFileReader.isHG = True

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
dqmEnvSiPixelAliHG = DQMEDHarvester('DQMHarvestingMetadata',
                                  subSystemFolder = cms.untracked.string('AlCaReco'),  
                                  )

ALCAHARVESTSiPixelAliHG = cms.Sequence(SiPixelAliMilleFileExtractorHG*
                                     SiPixelAliPedeAlignmentProducerHG*
                                     SiPixelAliDQMModuleHG*
                                     dqmEnvSiPixelAliHG)
