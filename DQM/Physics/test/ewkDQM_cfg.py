import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkDQM")
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
        '/store/relval/CMSSW_3_1_0/RelValZEE/GEN-SIM-RECO/MC_31X_V1-v1/0001/E0B5B4FD-C066-DE11-AB1E-001D09F24600.root',
        '/store/relval/CMSSW_3_1_0/RelValZEE/GEN-SIM-RECO/MC_31X_V1-v1/0001/B0E8F011-C366-DE11-A49F-001D09F2516D.root',
        '/store/relval/CMSSW_3_1_0/RelValZEE/GEN-SIM-RECO/MC_31X_V1-v1/0001/6A7A4717-C166-DE11-83FF-001D09F24FEC.root',
        '/store/relval/CMSSW_3_1_0/RelValZEE/GEN-SIM-RECO/MC_31X_V1-v1/0001/58E1635B-DE66-DE11-9CDD-0019B9F6C674.root'
#        '/store/relval/CMSSW_3_1_0/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/D22D3E9C-8966-DE11-900A-001617C3B66C.root',
#        '/store/relval/CMSSW_3_1_0/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/7E8944E8-8E66-DE11-9BBF-001D09F23A84.root',
#        '/store/relval/CMSSW_3_1_0/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/384836AE-D166-DE11-8D68-001D09F2983F.root',
#        '/store/relval/CMSSW_3_1_0/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/04E84AF4-8366-DE11-BC25-001D09F28D4A.root'
#        '/store/relval/CMSSW_3_1_0/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/A20BA375-DE66-DE11-BD52-000423D986A8.root',
#        '/store/relval/CMSSW_3_1_0/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/9CAA54AA-7366-DE11-993B-001617E30D52.root',
#        '/store/relval/CMSSW_3_1_0/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/1A82BC48-7C66-DE11-8007-001D09F28D4A.root',
#        '/store/relval/CMSSW_3_1_0/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/0A9770A5-7A66-DE11-819D-001D09F29538.root'


    )
)
#process.MessageLogger = cms.Service("MessageLogger",
#    destinations = cms.untracked.vstring('detailedInfo',
#        'cout')
#)
#process.ana = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.ewkDQM+process.dqmSaver)

