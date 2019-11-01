import FWCore.ParameterSet.Config as cms
process = cms.Process("ExoticaDQM")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.GlobalTag.globaltag = cms.string("80X_mcRun2_asymptotic_v13")

process.load("DQM.Physics.ExoticaDQM_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.dqmSaver.workflow = cms.untracked.string('/Physics/Exotica/TEST')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
       #'/store/relval/CMSSW_8_0_1/JetHT/GEN-SIM-RECO/80X_dataRun2_relval_v3_rereco_RelVal_jetHT2015DReReco-v1/10000/FA5A1EA3-73ED-E511-9B51-0025905B8582.root',
       #'/store/data/Run2011A/MinimumBias/RAW/v1/000/165/121/0699429A-B37F-E011-A57A-0019B9F72D71.root',
       '/store/relval/CMSSW_8_1_0_pre12/RelValDisplacedSUSY_stopToBottom_M_300_1000mm_13/GEN-SIM-RECO/81X_mcRun2_asymptotic_v8-v1/00000/BE4178D3-9A86-E611-8B45-0CC47A7C3450.root',
       #'/store/relval/CMSSW_8_0_1/RelValTTbar_13/GEN-SIM-DIGI-RECO/80X_mcRun2_asymptotic_v6_FastSim-v1/10000/0402D6BF-BAE4-E511-B293-00248C55CC7F.root',
    )
)

process.load('JetMETCorrections.Configuration.JetCorrectors_cff')

process.p = cms.Path( process.ExoticaDQM + process.dqmSaver)
#process.p = cms.Path( process.dqmAk4PFCHSL1FastL2L3CorrectorChain + process.ExoticaDQM + process.dqmSaver)
