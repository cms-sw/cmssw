import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO3")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_3XY_V10::All')


process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
### For 219, file from RelVal
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V10-v1/0003/0AC2D5EB-0FBE-DE11-ABF3-001A92971B04.root')
)

#process.myjetplustrack = cms.EDFilter("JetPlusTrackAnalysis",
#    HistOutFile = cms.untracked.string('JetAnalysis.root'),
#    src2 = cms.InputTag("iterativeCone5GenJets"),
#    src3 = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
#    src4 = cms.InputTag("ZSPJetCorJetIcone5"),
#    src1 = cms.InputTag("iterativeCone5CaloJets"),
#    Cone = cms.double(0.5),
#    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
#    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
#    HORecHitCollectionLabel = cms.InputTag("horeco"),
#    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
#    inputTrackLabel = cms.untracked.string('generalTracks')
#)

process.myjetplustrack = cms.EDFilter("JetPlusTrackAnalysis",
    HistOutFile = cms.untracked.string('JetAnalysis.root'),
    src2 = cms.InputTag("sisCone5GenJets"),
    src3 = cms.InputTag("JetPlusTrackZSPCorJetSiscone5"),
    src4 = cms.InputTag("ZSPJetCorJetSiscone5"),
    src1 = cms.InputTag("sisCone5CaloJets"),
    Cone = cms.double(0.5),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
    HORecHitCollectionLabel = cms.InputTag("horeco"),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
    inputTrackLabel = cms.untracked.string('generalTracks')
)



process.p1 = cms.Path(process.ZSPJetCorrectionsSisCone5*process.JetPlusTrackCorrectionsSisCone5*process.ZSPJetCorrectionsIcone5*process.JetPlusTrackCorrectionsIcone5*process.ZSPJetCorrectionsAntiKt5*process.JetPlusTrackCorrectionsAntiKt5*process.myjetplustrack)
