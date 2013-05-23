import FWCore.ParameterSet.Config as cms

process = cms.Process("dqm")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.load("HLTriggerOffline.Egamma.EgammaValidationReco_cff")
process.post=cms.EDAnalyzer("EmDQMPostProcessor",
                            subDir = cms.untracked.string("HLT/HLTEgammaValidationReco")                   
    )
process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            fileNames = cms.untracked.vstring(
       ## '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/3AFB84AF-3CE2-DE11-98A5-001D09F28EA3.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/3AE3CBE5-54E2-DE11-8438-003048D37514.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/3ABD83BC-4BE2-DE11-A9BB-000423D987E0.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/36C0F280-55E2-DE11-9A4C-000423D987FC.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/365C7833-54E2-DE11-8B07-003048D37514.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/343887ED-3DE2-DE11-B557-000423D9989E.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/32B30833-35E2-DE11-A4C1-003048D3756A.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/309101CD-39E2-DE11-A817-003048D373AE.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/307E47BA-4BE2-DE11-B36D-000423D987FC.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/2C8FF376-53E2-DE11-B626-003048D37456.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/2C7C14F0-4EE2-DE11-BEF7-000423D94534.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/2C15506A-32E2-DE11-A2DD-0030486730C6.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/2ACE8122-46E2-DE11-B97D-001617E30D12.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/2A34B7EE-3DE2-DE11-A4BD-00304879FA4A.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/281E7A4A-5AE2-DE11-9824-0030487D0D3A.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/2237806B-48E2-DE11-A87E-001D09F2512C.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/1E7A09CC-42E2-DE11-BEB9-000423D94C68.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/1E50FE73-57E2-DE11-ABB5-001617C3B6E2.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/1E47EDD7-49E2-DE11-B7E2-003048D2C1C4.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/1E2B66CD-42E2-DE11-AAF1-000423DD2F34.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/1C243E8D-3BE2-DE11-9917-001D09F24353.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/189EB1AB-4CE2-DE11-968F-001D09F24934.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/16EA3C34-35E2-DE11-A674-0030487A18F2.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/146D9B6F-47E2-DE11-BB59-0030487A18F2.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/10C5CCF1-3FE2-DE11-A4B6-000423D999CA.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/102E572A-41E2-DE11-BD6B-000423D99658.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/0CF61965-57E2-DE11-AF2C-003048D2C108.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/0C84E124-38E2-DE11-BF58-003048D3750A.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/08762134-5AE2-DE11-93B0-000423D99614.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/0816F593-36E2-DE11-B4DB-0019B9F70468.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/063F1306-39E2-DE11-BFC7-003048D37538.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/02C42DDD-43E2-DE11-9760-000423D98BC4.root',
##        '/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v2/000/123/596/02C28F6F-47E2-DE11-940A-0030487A1FEC.root'
    
##  'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_1.root',
## 'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_10.root',
## 'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_2.root',
## 'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_3.root',
## 'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_4.root',
## 'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_5.root',
## 'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_6.root',
## 'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_7.root',
## 'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_8.root',
## 'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_9.root'
       'file:./EGMFirstCollisionSkimHLT.root'
                                             )
                            )

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.p = cms.EndPath(process.post+process.dqmSaver)

process.testW = cms.Path(process.egammaValidationSequenceReco)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
