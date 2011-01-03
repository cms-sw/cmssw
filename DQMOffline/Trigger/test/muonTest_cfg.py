import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonOfflineAnalysis")

process.load("DQMOffline.Trigger.MuonOffline_Trigger_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_11_0_pre2/RelValZMM/GEN-SIM-RECO/START310_V3-v1/0058/6670ED8D-DA14-E011-9C6F-0026189437E8.root',
        '/store/relval/CMSSW_3_11_0_pre2/RelValZMM/GEN-SIM-RECO/START310_V3-v1/0055/DC47F951-7714-E011-BE46-001A92971BBE.root',
        '/store/relval/CMSSW_3_11_0_pre2/RelValZMM/GEN-SIM-RECO/START310_V3-v1/0055/2EA2784A-7114-E011-802E-001A92971BBE.root',
        '/store/relval/CMSSW_3_11_0_pre2/RelValZMM/GEN-SIM-RECO/START310_V3-v1/0055/0035ACDF-7A14-E011-B15B-001A92810AB2.root',
    ),
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
    fileName = cms.untracked.string('muonTest.root'),
)

process.analyzerpath = cms.Path(
    process.muonFullOfflineDQM *
    process.MEtoEDMConverter # *
    # process.dqmStoreStats
)

process.outpath = cms.EndPath(process.out)
