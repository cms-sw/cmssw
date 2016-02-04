import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTPOST")


process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0009/D0168338-48B7-DE11-B0CB-001D09F295A1.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0009/CE27D0D4-45B7-DE11-9E26-001D09F251FE.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0009/7872D05D-4AB7-DE11-8511-001D09F231B0.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0009/2622FDC7-46B7-DE11-8DC3-000423D6006E.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0009/0E10311B-76B7-DE11-B782-000423D944FC.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0002/F2F4D306-74B7-DE11-B1F3-001731AF67BF.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0002/DA5D256D-C5B7-DE11-BF58-003048678BAA.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0002/B0C96355-71B7-DE11-BE0F-001A92810ABA.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0002/8C7C0CFC-79B7-DE11-9461-001A92810ADE.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0002/8A0239B3-76B7-DE11-A0C4-001731A28857.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0002/72F2EDE6-6FB7-DE11-8998-003048D15E02.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0001/FAFDB886-6CB7-DE11-8D3D-001731AF65E9.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0001/7068CCED-6BB7-DE11-A6B4-001731AF6A89.root',
                '/store/relval/CMSSW_3_3_0/RelValZTT/GEN-SIM-RECO/STARTUP31X_V8-v1/0001/10A7BEEA-6BB7-DE11-986F-0018F3D09676.root'
        
    )
)


process.load("FWCore.MessageService.MessageLogger_cfi")
process.DQMStore = cms.Service("DQMStore")

#Load DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#Reconfigure Environment and saver
#process.dqmEnv.subSystemFolder = cms.untracked.string('HLT/HLTTAU')
#process.DQM.collectorPort = 9091
#process.DQM.collectorHost = cms.untracked.string('pcwiscms10')

process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.workflow = cms.untracked.string('/A/N/C')
process.dqmSaver.forceRunNumber = cms.untracked.int32(123)

#Load Converter
process.load("DQMServices.Components.EDMtoMEConverter_cff")


#Load The Post processor
process.load("HLTriggerOffline.Tau.Validation.HLTTauPostValidation_cfi")
process.load("HLTriggerOffline.Tau.Validation.HLTTauQualityTests_cff")


#Define the Paths

process.postProcess = cms.EndPath(process.EDMtoMEConverter+process.HLTTauPostVal+process.hltTauRelvalQualityTests+process.dqmSaver)



