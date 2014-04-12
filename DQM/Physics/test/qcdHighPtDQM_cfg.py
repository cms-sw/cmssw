import FWCore.ParameterSet.Config as cms

process = cms.Process("QcdHighPtDQM")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.dqmSaver.workflow = cms.untracked.string('/Physics/QCDPhysics/Jets')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_50_80/GEN-SIM-RECO/STARTUP_30X_v1/0002/E827B5EF-FE16-DE11-8C9F-00304867C136.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_50_80/GEN-SIM-RECO/STARTUP_30X_v1/0002/9C49DDC4-5C18-DE11-96DB-001A92811736.root'
                           )
                    )

process.QcdHighPtDQM = cms.EDAnalyzer("QcdHighPtDQM",
                                    jetTag = cms.untracked.InputTag("sisCone5CaloJets"),
                                    metTag1 = cms.untracked.InputTag("met"),
                                    metTag2 = cms.untracked.InputTag("metHO"),
                                    metTag3 = cms.untracked.InputTag("metNoHF"),
                                    metTag4 = cms.untracked.InputTag("metNoHFHO")
                                     
)
process.p = cms.Path(process.QcdHighPtDQM+process.dqmSaver)

