import FWCore.ParameterSet.Config as cms
process = cms.Process("ExoticaDQM")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.GlobalTag.globaltag = cms.string("80X_mcRun2_asymptotic_v4")

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
        '/store/relval/CMSSW_8_0_0_pre6/RelValDisplacedSUSY_stopToBottom_M_300_1000mm_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1/10000/1659FC04-D7D0-E511-988B-0CC47A4D76C0.root',
        '/store/relval/CMSSW_8_0_0_pre6/RelValDisplacedSUSY_stopToBottom_M_300_1000mm_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1/10000/70F265BE-D9D0-E511-BE0C-0CC47A4C8F18.root',
        '/store/relval/CMSSW_8_0_0_pre6/RelValDisplacedSUSY_stopToBottom_M_300_1000mm_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1/10000/D2841570-D9D0-E511-B134-0025905A60A6.root',        
       )
)
    
process.load('JetMETCorrections.Configuration.JetCorrectors_cff')

process.p = cms.Path( process.ak4PFCHSL1FastL2L3CorrectorChain + process.ExoticaDQM + process.dqmSaver)
