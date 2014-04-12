import FWCore.ParameterSet.Config as cms

process = cms.Process("CORRECTIONS")

#process.load("JetMETCorrections.Configuration.MCJetCorrections152_cff")

process.load("JetMETCorrections.Configuration.L7PartonCorrections_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/086FE832-2686-DD11-BD82-001617C3B76E.root')
)

process.partonJetCorrectionExample = cms.EDFilter("PartonJetCorrectionExample",
    src = cms.InputTag("hltMCJetCorJetIcone5"),
    gJetCorrector = cms.string('L7PartonJetCorrectorIC5gJet'),
    qJetCorrector = cms.string('L7PartonJetCorrectorIC5qJet'),
    bJetCorrector = cms.string('L7PartonJetCorrectorIC5bJet'),
    bTopCorrector = cms.string('L7PartonJetCorrectorIC5bTop')
)

#process.p = cms.Path(process.MCJetCorJetIcone5*process.partonJetCorrectionExample)
process.p = cms.Path(process.partonJetCorrectionExample)


