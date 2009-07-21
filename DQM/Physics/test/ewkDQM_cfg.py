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
        '/store/relval/CMSSW_3_1_0_pre9/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0007/982B860E-F64E-DE11-85DE-001617E30CC8.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0007/821CB3E3-F54E-DE11-91AE-001D09F2438A.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0007/34A39AFA-F54E-DE11-B847-001617DC1F70.root'
    )
)

process.p = cms.Path(process.ewkDQM+process.dqmSaver)

