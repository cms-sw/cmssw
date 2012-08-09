import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


process = cms.Process("HLTMuonOfflineAnalysis")

#### Load packages
process.load("DQMOffline.Trigger.MuonOffline_Trigger_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")

#### Process command-line arguments
options = VarParsing('analysis')
options.setDefault('inputFiles', 'file:/shome/pablom/DQM/CMSSW_5_2_3/src/DQMOffline/Trigger/test/multicrab/B28A65E6-01BA-E111-B6BB-003048F118D2.root')
options.setDefault('outputFile', './muonTest.root')

options.parseArguments()


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "START53_V4::All"



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(options.inputFiles),
)

process.DQMStore = cms.Service("DQMStore")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    categories = cms.untracked.vstring('HLTMuonVal'),
    destinations = cms.untracked.vstring('cout')
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_MEtoEDMConverter_*_*'
    ),
    fileName = cms.untracked.string(options.outputFile),
)

process.analyzerpath = cms.Path(
    process.muonFullOfflineDQM *
    process.MEtoEDMConverter # *
    # process.dqmStoreStats
)

process.outpath = cms.EndPath(process.out)
