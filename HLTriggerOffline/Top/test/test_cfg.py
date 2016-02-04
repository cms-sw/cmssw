import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load("HLTriggerOffline.Top.topvalidation_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")




process.source = cms.Source("PoolSource",
    #AlCaReco File
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/CC80B73A-CA57-DE11-BC2F-000423D99896.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/C68B7F1A-CD57-DE11-B706-00304879FA4A.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/9CA9BBC1-CD57-DE11-B62D-001D09F2424A.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/88AD5382-C657-DE11-831F-001D09F24498.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/7C7CDD0F-C457-DE11-8EEE-000423D951D4.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/4C30BDFF-B657-DE11-907A-001D09F24600.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/383036B6-0458-DE11-819F-001D09F29524.root'

    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)


#process.FEVT = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
#    fileName = cms.untracked.string('reco-gr-dqm.root')
#)

#process.p = cms.Path(process.muonAlignment*process.MEtoEDMConverter)

process.load("DQMServices.Core.DQM_cfg")

process.dqmSaverMy = cms.EDAnalyzer("DQMFileSaver",
        convention=cms.untracked.string("Offline"),
    
          workflow=cms.untracked.string("/HLT/Top/Validation"),
        
         dirName=cms.untracked.string("."),
         saveAtJobEnd=cms.untracked.bool(True),                        
         forceRunNumber=cms.untracked.int32(999871)
	)

process.p = cms.Path(process.HLTTopVal
	*process.dqmSaverMy
	)





