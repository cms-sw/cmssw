import FWCore.ParameterSet.Config as cms

process = cms.Process("DIJETS")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring('/store/relval/CMSSW_2_1_0/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0000/001EA63A-DF60-DD11-9D5A-001A92810AA6.root')
)

process.DiJProd = cms.EDProducer("AlCaDiJetsProducer",
    jetsInput = cms.InputTag("iterativeCone5CaloJets"),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    hbheInput = cms.InputTag("hbhereco"),
    hoInput = cms.InputTag("horeco"),
    hfInput = cms.InputTag("hfreco")
)

process.DiJRecos = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_DiJProd_*_*'),
    fileName = cms.untracked.string('/tmp/krohotin/dijets.root')
)

process.p = cms.Path(process.DiJProd)
process.e = cms.EndPath(process.DiJRecos)

