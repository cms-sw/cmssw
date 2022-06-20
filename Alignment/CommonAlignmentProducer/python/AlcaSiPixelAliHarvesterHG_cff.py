import FWCore.ParameterSet.Config as cms

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
from Alignment.MillePedeAlignmentAlgorithm.MillePedeDQMModule_cff import *

SiPixelAliPedeAlignmentProducerHG = AlignmentProducer.clone(
    ParameterBuilder = dict(
        Selector = cms.PSet(
            alignParams = cms.vstring(
                "TrackerP1PXBLadder,111111",
                "TrackerP1PXECPanel,111111",
            )
        )
    ),
    doMisalignmentScenario = False,
    checkDbAlignmentValidity = False,
    applyDbAlignment = True,
    tjTkAssociationMapTag = 'TrackRefitter2',
    saveToDB = True,
    trackerAlignmentRcdName = "TrackerAlignmentHGRcd"
)

SiPixelAliPedeAlignmentProducerHG.algoConfig = MillePedeAlignmentAlgorithm.clone(
    mode = 'pede',
    runAtPCL = True,
    mergeBinaryFiles = [SiPixelAliMilleFileExtractorHG.outputBinaryFile.value()],
    binaryFile = '',
    TrajectoryFactory = cms.PSet(BrokenLinesTrajectoryFactory),
    minNumHits = 10,
    fileDir = 'HGalignment/',
    pedeSteerer = dict(
        pedeCommand = 'pede',
        method = 'inversion  5  0.8',
        options = cms.vstring(
            #'regularisation 1.0 0.05', # non-stated pre-sigma 50 mrad or 500 mum
            'entries 500',
            'chisqcut  30.0  4.5',
            'threads 1 1',
            'closeandreopen',
            'skipemptycons' 
            #'outlierdownweighting 3','dwfractioncut 0.1'
            #'outlierdownweighting 5','dwfractioncut 0.2'
        ),
        fileDir = 'HGalignment/',
        runDir = cms.untracked.string('HGalignment/'),
        steerFile = 'pedeSteerHG',
        pedeDump = 'pedeHG.dump'
    ),
    pedeReader = dict(
        fileDir = 'HGalignment/'
    ),
    MillePedeFileReader = dict(
        fileDir = "HGalignment/",
        isHG = True
    )
)

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
