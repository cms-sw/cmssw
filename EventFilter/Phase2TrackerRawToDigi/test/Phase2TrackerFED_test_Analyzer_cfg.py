import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("OWNPARTICLES")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

# Input source
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring( 'file:'+sys.argv[-1])
)

#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'GR_R_42_V14::All'

process.load('EventFilter.Phase2TrackerRawToDigi.test.Phase2TrackerFED_test_Analyzer_cfi')

# process.myProducerLabel = cms.EDProducer(
#     'Phase2TrackerExcludedFEDListProducer',
#     ProductLabel = cms.InputTag("source")
# )

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
)

  
process.p = cms.Path(process.FED_test_Analyzer)

process.e = cms.EndPath(process.out)
