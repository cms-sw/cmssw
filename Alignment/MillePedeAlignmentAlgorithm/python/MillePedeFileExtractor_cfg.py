import FWCore.ParameterSet.Config as cms

process = cms.Process("MillePedeFileExtractor")

process.load("FWCore.MessageService.MessageLogger_cfi")

# This is just a test configuration. It should not be loaded directly in any
# other configuration.
# The filenames below are just suggestions.
# To get all info about this module, type:
# edmPluginHelp -p MillePedeFileExtractor

# Using the normal standard messagelogger, with its standard configuration,
# but setting the category of messages to MillePedeFileActions
process.MessageLogger = process.MessageLogger.clone(
        categories = cms.untracked.vstring('MillePedeFileActions'),
        )

# Limit our test to 5 events (we work on run level anyway)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        ##'file:/afs/cern.ch/work/c/cerminar/pcl/ali/frombroen/new/CMSSW_7_4_4/src/PromptCalibProdSiPixelAli.root',
        ##'file:/afs/cern.ch/work/c/cerminar/pcl/ali/frombroen/new/CMSSW_7_4_4/src/PromptCalibProdSiPixelAli_2.root'
        ##'file:output.root',
        'file:PromptCalibProdSiPixelAli1.root',
        'file:PromptCalibProdSiPixelAli2.root',
    )
)

from Alignment.MillePedeAlignmentAlgorithm.millePedeFileExtractor_cfi import millePedeFileExtractor
process.testMillePedeFileExtractor = millePedeFileExtractor.clone(
    #FIXME: handle with an InputLabel instead of 
    #TODO: Above sentence needs to be finished, otherwise I don't know what to fix.
    fileBlobInputTag = cms.InputTag("SiPixelAliMillePedeFileConverter",""),
    # You can add formatting directives like "%04d" in the output file name to setup numbering.
    outputBinaryFile = cms.string('pippo%04d.dat'),
    fileDir = cms.string('/afs/cern.ch/work/b/bvanbesi/private/PCLALI/CMSSW_7_4_4/src/'))
    ##fileDir = cms.string('/afs/cern.ch/work/c/cerminar/pcl/ali/frombroen/new/CMSSW_7_4_4/src/'))

process.p = cms.Path(process.testMillePedeFileExtractor)
