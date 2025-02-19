import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Core.DQMStore_cfg")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource",

)

process.demo = cms.EDAnalyzer('StripValidationPlots',
                              inputFilename  = cms.untracked.string('../HotStripsOccupancy_140331.root'),
                              outputFilename = cms.untracked.string('StripValidationPlots.root')
)


process.p = cms.Path(process.demo)
