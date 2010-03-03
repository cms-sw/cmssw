import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO3")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V8::All')


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
fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_3_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0002/2254715C-A99B-DE11-9CA8-0018F3D096BA.root')
)

#process.myjetplustrack = cms.EDAnalyzer("JetPlusTrackAnalysis",
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

process.myjetplustrack = cms.EDAnalyzer("JetPlusTrackAnalysis",
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
