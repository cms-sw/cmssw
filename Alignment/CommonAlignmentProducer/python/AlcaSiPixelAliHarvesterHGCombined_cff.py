import FWCore.ParameterSet.Config as cms

SiPixelAliMilleFileExtractorHGMinBias = cms.EDAnalyzer("MillePedeFileExtractor",
    fileBlobInputTag = cms.InputTag("SiPixelAliMillePedeFileConverterHG",''),
    fileDir = cms.string('HGCombinedAlignment/'),
    # File names the Extractor will use to write the fileblobs in the root
    # file as real binary files to disk, so that the pede step can read them.
    # This includes the formatting directive "%04d" which will be expanded to
    # 0000, 0001, 0002,...
    outputBinaryFile = cms.string('pedeBinaryHGMinBias%04d.dat'))

SiPixelAliMilleFileExtractorHGZMuMu = cms.EDAnalyzer("MillePedeFileExtractor",
    fileBlobInputTag = cms.InputTag("SiPixelAliMillePedeFileConverterHGDimuon",''),
    fileDir = cms.string('HGCombinedAlignment/'),
    # File names the Extractor will use to write the fileblobs in the root
    # file as real binary files to disk, so that the pede step can read them.
    # This includes the formatting directive "%04d" which will be expanded to
    # 0000, 0001, 0002,...
    outputBinaryFile = cms.string('pedeBinaryHGDiMuon%04d.dat'))

from Alignment.MillePedeAlignmentAlgorithm.MillePedeAlignmentAlgorithm_cfi import *
from Alignment.CommonAlignmentProducer.AlignmentProducerAsAnalyzer_cff import AlignmentProducer
from Alignment.MillePedeAlignmentAlgorithm.MillePedeDQMModule_cff import *

SiPixelAliPedeAlignmentProducerHGCombined = AlignmentProducer.clone(
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
    trackerAlignmentRcdName = "TrackerAlignmentHGCombinedRcd"
)

SiPixelAliPedeAlignmentProducerHGCombined.algoConfig = MillePedeAlignmentAlgorithm.clone(
    mode = 'pede',
    runAtPCL = True,
    #mergeBinaryFiles = [SiPixelAliMilleFileExtractorHGMinBias.outputBinaryFile.value()],
    #mergeBinaryFiles = [SiPixelAliMilleFileExtractorHGZMuMu.outputBinaryFile.value()],
    mergeBinaryFiles = ['pedeBinaryHGMinBias%04d.dat','pedeBinaryHGDiMuon%04d.dat -- 10.0'],
    binaryFile = '',
    TrajectoryFactory = cms.PSet(BrokenLinesTrajectoryFactory),
    minNumHits = 10,
    fileDir = 'HGCombinedAlignment/',
    pedeSteerer = dict(
        pedeCommand = 'pede',
        method = 'inversion  5  0.8',
        options = [
            #'regularisation 1.0 0.05', # non-stated pre-sigma 50 mrad or 500 mum
            'entries 500',
            'chisqcut  30.0  4.5',
            'threads 1 1',
            'closeandreopen',
            'skipemptycons' 
            #'outlierdownweighting 3','dwfractioncut 0.1'
            #'outlierdownweighting 5','dwfractioncut 0.2'
        ],
        fileDir = 'HGCombinedAlignment/',
        runDir = 'HGCombinedAlignment/',
        steerFile = 'pedeSteerHGCombined',
        pedeDump = 'pedeHGCombined.dump'
    ),
    pedeReader = dict(
        fileDir = 'HGCombinedAlignment/'
    ),
    MillePedeFileReader = dict(
        fileDir = "HGCombinedAlignment/",
        isHG = True
    )
)

SiPixelAliDQMModuleHGCombined = SiPixelAliDQMModule.clone()
SiPixelAliDQMModuleHGCombined.outputFolder = "AlCaReco/SiPixelAliHGCombined"
SiPixelAliDQMModuleHGCombined.MillePedeFileReader.fileDir = "HGCombinedAlignment/"
SiPixelAliDQMModuleHGCombined.MillePedeFileReader.isHG = True

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
dqmEnvSiPixelAliHGCombined = DQMEDHarvester('DQMHarvestingMetadata',
                                    subSystemFolder = cms.untracked.string('AlCaReco'))

ALCAHARVESTSiPixelAliHGCombined = cms.Sequence(SiPixelAliMilleFileExtractorHGMinBias*
                                               SiPixelAliMilleFileExtractorHGZMuMu*
                                               SiPixelAliPedeAlignmentProducerHGCombined*
                                               SiPixelAliDQMModuleHGCombined*
                                               dqmEnvSiPixelAliHGCombined)
-- dummy change --
