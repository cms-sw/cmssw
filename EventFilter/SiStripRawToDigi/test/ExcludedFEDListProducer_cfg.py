import FWCore.ParameterSet.Config as cms

process = cms.Process("OWNPARTICLES")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        '/store/data/Run2010B/MinimumBias/RAW/v1/000/147/926/F4FCCEDB-3CD7-DF11-B220-001D09F28EC1.root'
    )
)

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR_R_42_V14::All'

process.load('EventFilter.SiStripRawToDigi.ExcludedFEDListProducer_cfi')

# process.myProducerLabel = cms.EDProducer(
#     'SiStripExcludedFEDListProducer',
#     ProductLabel = cms.InputTag("rawDataCollector")
# )

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
)

  
process.p = cms.Path(process.SiStripExcludedFEDListProducer)

process.e = cms.EndPath(process.out)
