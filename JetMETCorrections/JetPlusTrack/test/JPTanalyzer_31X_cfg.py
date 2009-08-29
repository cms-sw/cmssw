import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# Some generic services and conditions data
process.Timing = cms.Service("Timing")
process.Tracer = cms.Service("Tracer",sourceSeed = cms.untracked.string("$$"))
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP31X_V2::All')

# Input files: RelVal QCD 80-120 GeV, STARTUP conditions, 9000 events, from CMSSW_3_1_2
process.source = cms.Source(
    "PoolSource", 
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/DCAE40E8-CA78-DE11-8F20-001D09F2305C.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/D099AB85-CA78-DE11-9A5E-001D09F2503C.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/B2E190DF-CA78-DE11-9F35-001D09F2532F.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/90A07B7B-CA78-DE11-A4B7-001D09F26C5C.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/4C5F86D1-CB78-DE11-8650-000423D6B42C.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/2247873A-B378-DE11-8F5B-001D09F24664.root',
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

