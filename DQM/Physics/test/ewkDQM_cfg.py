import FWCore.ParameterSet.Config as cms
process = cms.Process("EwkDQM")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START52_V5::All'

process.load("DQM.Physics.ewkDQM_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/My/Test/DataSet')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_5_2_3/RelValWM/GEN-SIM-RECO/START52_V5-v1/0043/0C3EEBF0-2B7A-E111-A039-0018F3D0970C.root',
    '/store/relval/CMSSW_5_2_3/RelValWM/GEN-SIM-RECO/START52_V5-v1/0042/1E26FC68-EF79-E111-8FC6-001BFCDBD100.root',
    '/store/relval/CMSSW_5_2_3/RelValWM/GEN-SIM-RECO/START52_V5-v1/0042/9C0EAAEE-F079-E111-AB99-003048FFCBA8.root'
    )
)
#process.MessageLogger = cms.Service("MessageLogger",
#    destinations = cms.untracked.vstring('detailedInfo',
#        'cout')
#)
#process.ana = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.ewkDQM+process.dqmSaver)

