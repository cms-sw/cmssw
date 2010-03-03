import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO3")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_3XY_V14::All')

process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrectionsBG_cff")

process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")
process.ZSPJetCorJetIcone5.src = cms.InputTag("iterativeConePu5CaloJets")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
### For 219, file from RelVal
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0003/C25C472F-79ED-DE11-8F58-001D09F24600.root'
)
)

process.myjetplustrack = cms.EDAnalyzer("JetPlusTrackAnalysis",
    HistOutFile = cms.untracked.string('JetAnalysis.root'),
    src2 = cms.InputTag("iterativeCone5GenJets"),
    src3 = cms.InputTag("JetPlusTrackZSPCorJetIcone5BG"),
    src4 = cms.InputTag("ZSPJetCorJetIcone5"),
    src1 = cms.InputTag("iterativeConePu5CaloJets"),
    Cone = cms.double(0.5),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
    HORecHitCollectionLabel = cms.InputTag("horeco"),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
    inputTrackLabel = cms.untracked.string('hiGlobalPrimTracks')
)

process.rawtodigi_step = cms.Path(process.RawToDigi)
process.reco = cms.Path(process.reconstructionHeavyIons)
process.zsp = cms.Path(process.ZSPJetCorJetIcone5)
process.jptbg = cms.Path(process.JetPlusTrackCorrectionsIcone5BG)
process.myan = cms.Path(process.myjetplustrack)

process.schedule = cms.Schedule(process.rawtodigi_step,process.reco,process.zsp,process.jptbg,process.myan)

#process.schedule = cms.Schedule(process.rawtodigi_step,process.reco,process.zsp,process.myan)

#process.p1 = cms.Path(process.RawToDigi*process.reconstructionHeavyIons*process.ZSPJetCorJetIcone5*process.JetPlusTrackCorrectionsIcone5*process.JetPlusTrackCorrectionsIcone5BG*process.myjetplustrack)

#process.p1 = cms.Path(process.ZSPJetCorrectionsSisCone5*process.JetPlusTrackCorrectionsSisCone5*process.ZSPJetCorrectionsIcone5*process.JetPlusTrackCorrectionsIcone5*process.ZSPJetCorrectionsAntiKt5*process.JetPlusTrackCorrectionsAntiKt5*process.myjetplustrack)
