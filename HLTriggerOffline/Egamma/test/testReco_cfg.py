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
 'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_1.root',
'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_10.root',
'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_2.root',
'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_3.root',
'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_4.root',
'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_5.root',
'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_6.root',
'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_7.root',
'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_8.root',
'rfio:/castor/cern.ch/user/e/egrace/SingleElectron/SingleElectronPt40_cfi_GEN_SIM_DIGI_L1_HLT_9.root'
       
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
