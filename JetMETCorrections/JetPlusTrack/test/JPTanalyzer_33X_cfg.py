import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# Some generic services and conditions data
#process.Timing = cms.Service("Timing")
#process.Tracer = cms.Service("Tracer",sourceSeed = cms.untracked.string("$$"))
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP31X_V8::All')

# Input files: RelVal QCD 80-120 GeV, STARTUP conditions, 9000 events, from CMSSW_3_2_5 (replace with 33X when available!)
process.source = cms.Source(
    "PoolSource", 
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_3_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V8-v2/0000/D8D6F277-C5C7-DE11-A59F-002618943962.root',
    '/store/relval/CMSSW_3_3_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V8-v2/0000/C8072F59-59C8-DE11-BB0A-00261894393B.root',
    '/store/relval/CMSSW_3_3_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V8-v2/0000/BC410BBC-C4C7-DE11-BA4F-002618FDA237.root',
    '/store/relval/CMSSW_3_3_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V8-v2/0000/32E84E16-C4C7-DE11-A181-002618943833.root',
    '/store/relval/CMSSW_3_3_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V8-v2/0000/0295FB14-C4C7-DE11-834B-002618943862.root',
    ),
    )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

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
from RecoJets.JetProducers.ic5GenJets_cfi import iterativeCone5GenJets
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
process.load("JetMETCorrections.Configuration.ZSPJetCorrections31X_cff")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

# Analyzer module
process.myanalysis = cms.EDAnalyzer(
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

