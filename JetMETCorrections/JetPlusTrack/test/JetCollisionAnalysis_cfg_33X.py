import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO3")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.globaltag = cms.string('GR09_P_V7::All')


process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
### For 219, file from RelVal
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/596/F494AB9A-40E2-DE11-8D1E-000423D33970.root')
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

process.myjetplustrack = cms.EDFilter("JetPlusTrackCollisionAnalysis",
    HistOutFile = cms.untracked.string('JetAnalysis.root'),
    src1 = cms.InputTag("ak5CaloJets"),
    src2 = cms.InputTag("ZSPJetCorJetAntiKt5"),
    src3 = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    Cone = cms.double(0.5),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
#    HORecHitCollectionLabel = cms.InputTag("horeco"),
#    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
    HBHENZSRecHitCollectionLabel = cms.InputTag("hbherecoMB"),
    inputTrackLabel = cms.untracked.string('generalTracks')
)



process.p1 = cms.Path(process.ZSPJetCorrectionsAntiKt5*process.JetPlusTrackCorrectionsAntiKt5*process.myjetplustrack)
