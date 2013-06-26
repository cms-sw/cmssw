import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMOffline.Trigger.MuonPostProcessor_cff")
process.load("DQMOffline.Trigger.MuonHLTValidation_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('HLTMuonVal'),
    debugModules = cms.untracked.vstring('*'),
    threshold = cms.untracked.string('INFO'),
    HLTMuonVal = cms.untracked.PSet(
        #threshold = cms.untracked.string('DEBUG'),
        limit = cms.untracked.int32(100000),
    ),
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:muonTest.root'),
)

process.path = cms.Path(
    process.EDMtoME *
    process.hltMuonPostVal # *
    # process.muonHLTCertSeq *
    # process.dqmStoreStats
)

process.endpath = cms.EndPath(process.dqmSaver)
