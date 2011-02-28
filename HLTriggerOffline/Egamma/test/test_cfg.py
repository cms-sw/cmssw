import FWCore.ParameterSet.Config as cms

process = cms.Process("dqm")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("HLTriggerOffline.Egamma.EgammaValidation_cff")
process.post=cms.EDAnalyzer("EmDQMPostProcessor",
                            subDir = cms.untracked.string("HLT/HLTEgammaValidation")                   
    )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0010/D4F23B39-22EE-DE11-B0A1-0026189438D6.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/F0F83DF7-86ED-DE11-9BBB-002618943913.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/DAB03EB6-84ED-DE11-B977-0026189438A2.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/CC39E03E-85ED-DE11-90B1-002618943807.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/ACFE3EF8-86ED-DE11-AE8B-002618943951.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/969677A8-85ED-DE11-855E-003048678F8E.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/6EAB39FD-86ED-DE11-A95C-002354EF3BDA.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/6CEB982F-85ED-DE11-AA62-00261894380D.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/5EE521FB-85ED-DE11-AEC0-00304867C1B0.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/5C27B202-86ED-DE11-8033-003048678B1A.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/32DEB35C-84ED-DE11-B5D8-002618943927.root',
                '/store/relval/CMSSW_3_5_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/1276CF45-85ED-DE11-A698-0026189438AE.root'
        

    )
                            )

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.p = cms.EndPath(process.post+process.dqmSaver)

process.testW = cms.Path(process.egammaValidationSequence)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
