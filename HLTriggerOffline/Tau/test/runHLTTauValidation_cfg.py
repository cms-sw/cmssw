import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTVAL")


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	'/store/relval/CMSSW_9_0_0/RelValQQH1352T_13/GEN-SIM-RECO/90X_mcRun2_asymptotic_v5-v1/00000/00BB2C00-480F-E711-9381-0CC47A7C3636.root',
	'/store/relval/CMSSW_9_0_0/RelValQQH1352T_13/GEN-SIM-RECO/90X_mcRun2_asymptotic_v5-v1/00000/2ED8B184-480F-E711-8E31-0CC47A4C8E28.root',
	'/store/relval/CMSSW_9_0_0/RelValQQH1352T_13/GEN-SIM-RECO/90X_mcRun2_asymptotic_v5-v1/00000/C8E5EDFA-480F-E711-994B-0025905B85CA.root'
    )
)

# Set GlobalTag (automatically)
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

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


#Load the Validation
process.load("HLTriggerOffline.Tau.Validation.HLTTauValidation_cff")

#Load The Post processor
process.load("HLTriggerOffline.Tau.Validation.HLTTauPostValidation_cfi")
process.load("HLTriggerOffline.Tau.Validation.HLTTauQualityTests_cff")


#Define the Paths
process.validation = cms.Path(process.HLTTauVal)

process.postProcess = cms.EndPath(process.HLTTauPostVal+process.hltTauRelvalQualityTests+process.dqmSaver)
#process.postProcess = cms.EndPath(process.dqmSaver)
process.schedule =cms.Schedule(process.validation,process.postProcess)



