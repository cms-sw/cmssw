import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# Some generic services and conditions data
process.Timing = cms.Service("Timing")
process.Tracer = cms.Service("Tracer",sourceSeed = cms.untracked.string("$$"))
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')

# Input files: RelVal QCD 80-120 GeV, 5000 events, from CMSSW_3_1_2
process.source = cms.Source(
    "PoolSource", 
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v2/0011/F605AE2E-D990-DE11-BFCB-001D09F2545B.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v2/0011/441FC5C8-4C91-DE11-BC78-000423D6BA18.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v2/0011/1210F6E7-D990-DE11-A327-001D09F2545B.root',
    ),
    )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

# Identify GenParticles to be used to build GenJets (ie, no neutrinos or BSM)
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.genParticlesForJets.ignoreParticleIDs = cms.vuint32(
    1000022, 2000012, 2000014,
    2000016, 1000039, 5000039,
    4000012, 9900012, 9900014,
    9900016, 39, 12, 14, 16
    )
process.genParticlesForJets.excludeFromResonancePids = cms.vuint32(12, 14, 16)

# Build reco::GenJets from GenParticles
from RecoJets.JetProducers.iterativeCone5GenJets_cff import iterativeCone5GenJets
process.iterativeCone5GenJetsNoNuBSM = iterativeCone5GenJets.clone()

# Jet-track association
from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import*
ZSPiterativeCone5JetTracksAssociatorAtVertex        = iterativeCone5JetTracksAssociatorAtVertex.clone() 
ZSPiterativeCone5JetTracksAssociatorAtVertex.jets   = cms.InputTag("ZSPJetCorJetIcone5")
ZSPiterativeCone5JetTracksAssociatorAtCaloFace      = iterativeCone5JetTracksAssociatorAtCaloFace.clone()
ZSPiterativeCone5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ZSPJetCorJetIcone5")
ZSPiterativeCone5JetExtender                        = iterativeCone5JetExtender.clone() 
ZSPiterativeCone5JetExtender.jets                   = cms.InputTag("ZSPJetCorJetIcone5")
ZSPiterativeCone5JetExtender.jet2TracksAtCALO       = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtCaloFace")
ZSPiterativeCone5JetExtender.jet2TracksAtVX         = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtVertex")
ZSPrecoJetAssociations = cms.Sequence(
    ZSPiterativeCone5JetTracksAssociatorAtVertex *
    ZSPiterativeCone5JetTracksAssociatorAtCaloFace *
    ZSPiterativeCone5JetExtender
    )

# ZSP and JPT corrections
process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

# Analyzer module
process.myanalysis = cms.EDFilter(
    "JPTAnalyzer",
    HistOutFile      = cms.untracked.string('analysis.root'),
    calojets         = cms.string('iterativeCone5CaloJets'),
    zspjets          = cms.string('ZSPJetCorJetIcone5'),
    genjets          = cms.string('iterativeCone5GenJetsNoNuBSM'),
    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorIcone5')
    )

# Path
process.p1 = cms.Path(
    process.genParticlesForJets *
    process.iterativeCone5GenJetsNoNuBSM *
    process.ZSPJetCorrections *
    process.ZSPrecoJetAssociations *
    process.myanalysis
    )

