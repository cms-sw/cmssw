import FWCore.ParameterSet.Config as cms

process = cms.Process("MillePedeFileExtractor")

process.load("FWCore.MessageService.MessageLogger_cfi")

# Using the normal standard messagelogger, with its standard configuration,
# but setting the category of messages to MillePedeFileActions
process.MessageLogger = process.MessageLogger.clone(
        categories = cms.untracked.vstring('MillePedeFileActions'),
        )

# Limit our test to 5 events (we work on run level anyway)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/work/c/cerminar/pcl/ali/frombroen/new/CMSSW_7_4_4/src/PromptCalibProdSiPixelAli.root',
        'file:/afs/cern.ch/work/c/cerminar/pcl/ali/frombroen/new/CMSSW_7_4_4/src/PromptCalibProdSiPixelAli_2.root'
    )
)

from Alignment.MillePedeFileExtractor.millePedeFileExtractor_cfi import millePedeFileExtractor
process.testMillePedeFileExtractor = millePedeFileExtractor.clone(
    #FIXME: handle with an InputLabel instead of 
    fileBlobModule = cms.string("SiPixelAliMillePedeFileConverter"),
    fileBlobLabel  = cms.string(''),
    outputBinaryFile = cms.string('pippo.dat'),
    fileDir = cms.string('/afs/cern.ch/work/c/cerminar/pcl/ali/frombroen/new/CMSSW_7_4_4/src/'))

process.p = cms.Path(process.testMillePedeFileExtractor)
