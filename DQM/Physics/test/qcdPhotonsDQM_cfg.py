import FWCore.ParameterSet.Config as cms

process = cms.Process("QcdPhotonsDQM")
process.load("DQM.Physics.qcdPhotonsDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/My/Test/DataSet')

## Geometry and Detector Conditions (needed for spike removal code)
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START38_V9::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_8_2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0018/B81EF896-9AAF-DF11-B31B-001A92971BCA.root',
    '/store/relval/CMSSW_3_8_2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0018/886F398A-B8AF-DF11-91A8-003048678FC6.root',
    '/store/relval/CMSSW_3_8_2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0018/7830F828-87AF-DF11-9DE0-003048678FD6.root',
    '/store/relval/CMSSW_3_8_2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0018/26CC6A78-A8AF-DF11-97A5-003048678F78.root',
    '/store/relval/CMSSW_3_8_2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0017/3E226F93-7FAF-DF11-A908-001A92810AF4.root'        
                           )
)

process.p = cms.Path(process.qcdPhotonsDQM+process.dqmSaver)

