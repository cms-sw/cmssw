import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

resolutionTest = DQMEDHarvester("DTResolutionTest",
    runningStandalone = cms.untracked.bool(True),
    diagnosticPrescale = cms.untracked.int32(1),
    calibModule = cms.untracked.bool(True),
    #Names of the quality tests: they must match those specified in "qtList"
    resDistributionTestName = cms.untracked.string('ResidualsDistributionGaussianTest'),
    meanTestName = cms.untracked.string('ResidualsMeanInRange'),
    sigmaTestName = cms.untracked.string('ResidualsSigmaInRange'),
    #Input/Output files
    readFile = cms.untracked.bool(False),
    inputFile = cms.untracked.string('/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/CRAFT/ttrig/DTkFactValidation_66722.root'),
    debug = cms.untracked.bool(False),
    OutputMEsInRootFile = cms.bool(True),
    OutputFileName = cms.string('provaDTkfactValidation2.root'),
    #Histo setting
    folderRoot = cms.untracked.string(''),
    histoTag2D = cms.untracked.string('hResDistVsDist_STEP3'),
    histoTag = cms.untracked.string('hResDist_STEP3'),
    STEP = cms.untracked.string('STEP3'),  
    meanMaxLimit =  cms.untracked.double(0.07),
    meanWrongHisto =  cms.untracked.bool(False),
    sigmaTest =  cms.untracked.bool(False),
    slopeTest =  cms.untracked.bool(False)
)
