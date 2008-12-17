import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO3")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
### For 219, file from RelVal
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/000AD2A4-6E86-DD11-AA99-000423D9863C.root')
)

process.myjetplustrack = cms.EDFilter("JetPlusTrackAnalysis",
    HistOutFile = cms.untracked.string('JetAnalysis.root'),
    src2 = cms.InputTag("iterativeCone5GenJets"),
    src3 = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
    src4 = cms.InputTag("ZSPJetCorJetIcone5"),
    src1 = cms.InputTag("iterativeCone5CaloJets"),
    Cone = cms.double(0.5),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
    HORecHitCollectionLabel = cms.InputTag("horeco"),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
    inputTrackLabel = cms.untracked.string('generalTracks')
)

process.p1 = cms.Path(process.ZSPJetCorrections*process.JetPlusTrackCorrections*process.myjetplustrack)
import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO3")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

process.load("JetMETCorrections.Configuration.ZSPJetCorrections152_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/000AD2A4-6E86-DD11-AA99-000423D9863C.root')
#    fileNames = cms.untracked.vstring('file:/tmp/kodolova/FC999068-DB60-DD11-9694-001A92971B16.root')
)

process.myjetplustrack = cms.EDFilter("JetPlusTrackAnalysis",
    HistOutFile = cms.untracked.string('JetAnalysis.root'),
    src2 = cms.InputTag("iterativeCone5GenJets"),
    src3 = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
    src4 = cms.InputTag("ZSPJetCorJetIcone5"),
    src1 = cms.InputTag("iterativeCone5CaloJets"),
    Cone = cms.double(0.5),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
    HORecHitCollectionLabel = cms.InputTag("horeco"),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
    inputTrackLabel = cms.untracked.string('generalTracks')
)

process.p1 = cms.Path(process.ZSPJetCorrections*process.JetPlusTrackCorrections*process.myjetplustrack)
import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO3")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

process.load("JetMETCorrections.Configuration.ZSPJetCorrections152_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/000AD2A4-6E86-DD11-AA99-000423D9863C.root')
#    fileNames = cms.untracked.vstring('file:/tmp/kodolova/FC999068-DB60-DD11-9694-001A92971B16.root')
)

process.myjetplustrack = cms.EDFilter("JetPlusTrackAnalysis",
    HistOutFile = cms.untracked.string('JetAnalysis.root'),
    src2 = cms.InputTag("iterativeCone5GenJets"),
    src3 = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
    src4 = cms.InputTag("ZSPJetCorJetIcone5"),
    src1 = cms.InputTag("iterativeCone5CaloJets"),
    Cone = cms.double(0.5),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
    HORecHitCollectionLabel = cms.InputTag("horeco"),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
    inputTrackLabel = cms.untracked.string('generalTracks')
)

process.p1 = cms.Path(process.ZSPJetCorrections*process.JetPlusTrackCorrections*process.myjetplustrack)
