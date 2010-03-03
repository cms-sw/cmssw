import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# Some generic services and conditions data
process.Timing = cms.Service("Timing")
process.Tracer = cms.Service("Tracer",sourceSeed = cms.untracked.string("$$"))

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V8::All')

# Input files: RelVal QCD 80-120 GeV, STARTUP conditions, 9000 events, from CMSSW_3_2_5 (replace with 33X when available!)
process.source = cms.Source(
    "PoolSource", 
    fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_3_3_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0002/D2C0C845-B09B-DE11-A6C6-001731AF66C2.root',
      '/store/relval/CMSSW_3_3_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0002/BCFF13A5-AF9B-DE11-849C-001A92810AD2.root',
      '/store/relval/CMSSW_3_3_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0002/7E7F3A97-AF9B-DE11-B338-001731AF68B9.root',
      '/store/relval/CMSSW_3_3_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0002/6C75CA2C-469C-DE11-AB6C-001731AF6847.root',
      '/store/relval/CMSSW_3_3_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0002/226D8597-AF9B-DE11-BF7D-001731AF6719.root')
    )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

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
# from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import*
# ZSPiterativeCone5JetTracksAssociatorAtVertex        = iterativeCone5JetTracksAssociatorAtVertex.clone() 
# ZSPiterativeCone5JetTracksAssociatorAtVertex.jets   = cms.InputTag("ZSPJetCorJetIcone5")
# ZSPiterativeCone5JetTracksAssociatorAtCaloFace      = iterativeCone5JetTracksAssociatorAtCaloFace.clone()
# ZSPiterativeCone5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ZSPJetCorJetIcone5")
# ZSPiterativeCone5JetExtender                        = iterativeCone5JetExtender.clone() 
# ZSPiterativeCone5JetExtender.jets                   = cms.InputTag("ZSPJetCorJetIcone5")
# ZSPiterativeCone5JetExtender.jet2TracksAtCALO       = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtCaloFace")
# ZSPiterativeCone5JetExtender.jet2TracksAtVX         = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtVertex")
# ZSPrecoJetAssociations = cms.Sequence(
#     ZSPiterativeCone5JetTracksAssociatorAtVertex *
#     ZSPiterativeCone5JetTracksAssociatorAtCaloFace *
#     ZSPiterativeCone5JetExtender
#     )

# ZSP and JPT corrections

process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

# Analyzer module
process.myanalysis = cms.EDAnalyzer(
    "JPTBjetAnalyzer",
    HistOutFile      = cms.untracked.string('bjetanalysis.root'),
#    calojets         = cms.string('iterativeCone5CaloJets'),
    calojets         = cms.string('sisCone5CaloJets'),
#    calojets         = cms.string('ak5CaloJets'),
#    zspjets          = cms.string('ZSPJetCorJetIcone5'),
    zspjets          = cms.string('ZSPJetCorJetSiscone5'),
#    zspjets          = cms.string('ZSPJetCorJetAntiKt5'),
#    genjets          = cms.string('iterativeCone5GenJetsNoNuBSM'),
    genjets          = cms.string('sisCone5GenJets'),
#    genjets          = cms.string('ak5GenJets'),
#    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorIcone5')
    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorSiscone5'),
#    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorAntiKt5')
    electrons = cms.string("gsfElectron"),
    muons = cms.string("muons"),                                 
    genparticles = cms.string("genParticles"),
    electron_pt_min = cms.double(0.),
    electron_abseta = cms.double(2.5),
    muon_pt_min = cms.double(0),
    muon_abseta = cms.double(2.5),  
    jet_pt_min = cms.double(20),
    jet_abseta = cms.double(2.5),     
    )

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

# Path
process.p1 = cms.Path(
    process.genParticlesForJets *
    process.iterativeCone5GenJetsNoNuBSM *
#    process.ZSPJetCorrectionsIcone5 *
#    process.ZSPrecoJetAssociationsIcone5 *
    process.ZSPJetCorrectionsSisCone5 *
    process.ZSPrecoJetAssociationsSisCone5 *
#    process.ZSPJetCorrectionsAntiKt5 *	
#    process.ZSPrecoJetAssociationsAntiKt5 *
    process.myanalysis 
    )

