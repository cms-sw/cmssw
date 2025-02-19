import FWCore.ParameterSet.Config as cms

process = cms.Process("DIJETS")


process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0001/087DC4B2-640A-DE11-86E5-000423D98DD4.root')
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

