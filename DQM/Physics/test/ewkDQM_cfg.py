import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkDQM")
process.load("DQM.Physics.ewkDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/My/Test/DataSet')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#        '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/FC71916C-756B-DE11-8631-000423D94700.root',
#        '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/B672A1C5-746B-DE11-93A8-000423D944F0.root',
#        '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/5A0F871C-756B-DE11-BFE1-001D09F2545B.root',
#        '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/44507D17-D66B-DE11-A165-000423D94524.root'
#        '/store/relval/CMSSW_3_1_1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/C8CEE598-CB6B-DE11-871F-001D09F2905B.root',
#        '/store/relval/CMSSW_3_1_1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/642F8176-C96B-DE11-9D10-000423D98BE8.root',
#        '/store/relval/CMSSW_3_1_1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/46AA6A11-D46B-DE11-A614-001D09F25438.root',
#        '/store/relval/CMSSW_3_1_1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/443BC1DD-CC6B-DE11-804C-000423D98EC4.root'
        '/store/relval/CMSSW_3_1_1/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v2/0003/08DBA390-F66B-DE11-A7EC-0030487A3C9A.root',
        '/store/relval/CMSSW_3_1_1/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/88B35291-C16B-DE11-B9AB-000423D98F98.root',
        '/store/relval/CMSSW_3_1_1/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/7A84CA74-BC6B-DE11-B2E1-000423D6CA6E.root',
        '/store/relval/CMSSW_3_1_1/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/50D84AA9-C46B-DE11-8209-000423D98BC4.root',
        '/store/relval/CMSSW_3_1_1/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/1E5726C8-E16B-DE11-A98B-000423D99AAE.root'
#        '/store/relval/CMSSW_3_1_1/RelValWM/GEN-SIM-RECO/STARTUP31X_V1-v2/0003/8AC3E97D-EF6B-DE11-ADBA-001D09F29619.root',
#        '/store/relval/CMSSW_3_1_1/RelValWM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/EA593FE4-E26B-DE11-8173-001D09F2438A.root',
#        '/store/relval/CMSSW_3_1_1/RelValWM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/9270F55B-E26B-DE11-994E-001D09F2AF1E.root',
#        '/store/relval/CMSSW_3_1_1/RelValWM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/8E5D0675-E36B-DE11-8F71-001D09F242EF.root'

    )
)
#process.MessageLogger = cms.Service("MessageLogger",
#    destinations = cms.untracked.vstring('detailedInfo',
#        'cout')
#)
#process.ana = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.ewkDQM+process.dqmSaver)

