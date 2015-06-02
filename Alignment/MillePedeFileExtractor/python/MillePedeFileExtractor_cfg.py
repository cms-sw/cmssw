import FWCore.ParameterSet.Config as cms

process = cms.Process("MillePedeFileExtractor")

process.load("FWCore.MessageService.MessageLogger_cfi")

# Using the normal standard messagelogger, with its standard configuration,
# but setting the category of messages to MillePedeFileActions
process.MessageLogger = process.MessageLogger.clone(
        categories = cms.untracked.vstring('MillePedeFileActions'),
        )

# Limit our test to 5 events (we work on run level anyway)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:myOutputFile.root'
    )
)

from Alignment.MillePedeFileExtractor.millePedeFileExtractor_cfi import millePedeFileExtractor
process.testMillePedeFileExtractor = millePedeFileExtractor.clone(
               fileBlobModule = cms.string("testMillePedeFileConverter"))

process.p = cms.Path(process.testMillePedeFileExtractor)
