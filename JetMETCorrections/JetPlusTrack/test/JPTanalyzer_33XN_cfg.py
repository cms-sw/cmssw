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
process.GlobalTag.globaltag = cms.string('STARTUP3X_V12::All')
# process.GlobalTag.globaltag = cms.string('MC_31X_V8::All')

# Input files: RelVal QCD 80-120 GeV, STARTUP conditions, 9000 events, from CMSSW_3_2_5 (replace with 33X when available!)
process.source = cms.Source(
    "PoolSource", 
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/FA7139E8-97BD-DE11-A3E2-002618943935.root',
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/BC3224A5-9ABD-DE11-A625-002354EF3BDB.root',
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/8C578DA3-C0BD-DE11-9DEA-0017312A250B.root',
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/7A29EA77-9DBD-DE11-A3BC-0026189438ED.root',
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/3EA8A506-10BE-DE11-BB21-0018F3D09704.root',
    '/store/relval/CMSSW_3_4_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/04383FF7-9EBD-DE11-8511-0018F3D09616.root',)
    )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

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

# process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")
process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

# Analyzer module
process.myanalysis = cms.EDFilter(
    "JPTAnalyzer",
    HistOutFile      = cms.untracked.string('analysis.root'),
    calojets         = cms.string('iterativeCone5CaloJets'),
#    calojets         = cms.string('sisCone5CaloJets'),
#    calojets         = cms.string('ak5CaloJets'),
    zspjets          = cms.string('ZSPJetCorJetIcone5'),
#    zspjets          = cms.string('ZSPJetCorJetSiscone5'),
#    zspjets          = cms.string('ZSPJetCorJetAntiKt5'),
    genjets          = cms.string('iterativeCone5GenJetsNoNuBSM'),
#    genjets          = cms.string('sisCone5GenJets'),
#    genjets          = cms.string('ak5GenJets'),
    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorIcone5')
#    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorSiscone5')
#    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorAntiKt5')
    )

process.dump = cms.EDFilter("EventContentAnalyzer")

# Path
process.p1 = cms.Path(
    process.genParticlesForJets *
    process.iterativeCone5GenJetsNoNuBSM *
    process.ZSPJetCorrectionsIcone5 *
    process.ZSPrecoJetAssociationsIcone5 *
#    process.ZSPJetCorrectionsSisCone5 *
#    process.ZSPrecoJetAssociationsSisCone5 *
#    process.ZSPJetCorrectionsAntiKt5 *	
#    process.ZSPrecoJetAssociationsAntiKt5 *
    process.myanalysis 
    )

