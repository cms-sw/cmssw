import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

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

# process.load("JetMETCorrections.Configuration.MCJetCorrections152_cff")
process.load("RecoJets.Configuration.GenJetParticles_cff")

# Make Cone8 Jets
from  RecoJets.JetProducers.iterativeCone5GenJets_cff import *
process.iterativeCone8GenJets = iterativeCone5GenJets.clone()
process.iterativeCone8GenJets.coneRadius = 0.8
process.iterativeCone8GenJets.alias = 'IC8GenJet'

from  RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
process.iterativeCone8CaloJets = iterativeCone5CaloJets.clone()
process.iterativeCone8CaloJets.coneRadius = 0.8
process.iterativeCone8CaloJets.alias = 'IC8CaloJet'

process.s1 = cms.Sequence((process.genParticlesForJets*process.iterativeCone8GenJets) + process.iterativeCone8CaloJets)

from JetMETCorrections.Configuration.ZSPJetCorrections152_cff import *
ZSPJetCorrectorIcone5.tagName = 'ZSPJetCorrectionsIC08_152'
ZSPJetCorJetIcone5.src = 'iterativeCone8CaloJets'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/000AD2A4-6E86-DD11-AA99-000423D9863C.root',
'/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/02D641CC-6D86-DD11-B1AA-001617C3B64C.root')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.myanalysis = cms.EDAnalyzer("AnalNHad",
    HistOutFile = cms.untracked.string('analysis_ZJet.root'),
    zspjets = cms.string('ZSPJetCorJetIcone5'),                                
    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorIcone5')
)

iterativeCone5JetTracksAssociatorAtVertex.jets = 'ZSPJetCorJetIcone5'
iterativeCone5JetTracksAssociatorAtCaloFace.jets = 'ZSPJetCorJetIcone5'
iterativeCone5JetExtender.jets = 'ZSPJetCorJetIcone5'
iterativeCone5JetExtender.coneSize = 0.8


process.p1 = cms.Path(process.s1*process.ZSPJetCorJetIcone5*recoJetAssociations*process.myanalysis)

# process.p1 = cms.Path(process.dump)
