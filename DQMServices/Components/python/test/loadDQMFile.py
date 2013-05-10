import FWCore.ParameterSet.Config as cms

process = cms.Process('DQMSTATS')

# import DQMStore service
process.load('DQMOffline.Configuration.DQMOffline_cff')

# actually read in the DQM ROOT file
process.load("DQMServices.Components.DQMFileReader_cfi")
process.dqmFileReader.FileNames =  cms.untracked.vstring("DQM_V0002_R000166841__Jet__Run2011A-PromptReco-v4__DQM.root")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Load actual Statistics package
process.load("DQMServices.Components.DQMStoreStats_cfi")
        
# Input source
process.source = cms.Source("EmptySource")

process.readFile = cms.Path(process.dqmFileReader)
process.stats    = cms.Path(process.dqmStoreStats)

# Schedule definition
process.schedule = cms.Schedule(process.readFile, process.stats)

