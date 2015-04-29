import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTVAL")


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#process.MessageLogger.categories.append("HLTTauDQMOffline")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_4_0_pre9/RelValHiggs200ChargedTaus_13/GEN-SIM-RECO/MCRUN2_74_V7-v1/00000/4AEEFAED-55D4-E411-A18E-0025905B855E.root',
       '/store/relval/CMSSW_7_4_0_pre9/RelValHiggs200ChargedTaus_13/GEN-SIM-RECO/MCRUN2_74_V7-v1/00000/6868EBC2-5CD4-E411-929B-0025905A6092.root',
       '/store/relval/CMSSW_7_4_0_pre9/RelValHiggs200ChargedTaus_13/GEN-SIM-RECO/MCRUN2_74_V7-v1/00000/F6859A13-E4D3-E411-8D59-0025905A6066.root'
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



