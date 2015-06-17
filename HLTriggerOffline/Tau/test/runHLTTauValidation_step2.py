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
        '/store/hidata/HIRun2011/HIMinBiasUPC/RAW/v1/000/181/913/06B6DB7E-C810-E111-AE7C-001D09F2305C.root'        
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



