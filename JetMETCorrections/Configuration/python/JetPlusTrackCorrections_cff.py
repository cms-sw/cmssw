import FWCore.ParameterSet.Config as cms

# File: JetCorrections.cff
# Author: O. Kodolova
# Date: 1/24/07
#
# Jet corrections with tracks for the icone5 jets with ZSP corrections.
# 
from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *
JetPlusTrackZSPCorrectorIcone5 = cms.ESSource("JetPlusTrackCorrectionService",
    JetTrackCollectionAtCalo = cms.InputTag("iterativeCone5JetTracksAssociatorAtCaloFace"),
    respalgo = cms.int32(5),
    JetTrackCollectionAtVertex = cms.InputTag("iterativeCone5JetTracksAssociatorAtVertex"),
    label = cms.string('JetPlusTrackZSPCorrectorIcone5')
)

JetPlusTrackZSPCorJetIcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("ZSPJetCorJetIcone5"),
    correctors = cms.vstring('JetPlusTrackZSPCorrectorIcone5'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetIcone5')
)

JetPlusTrackCorrections = cms.Sequence(recoJetAssociations*JetPlusTrackZSPCorJetIcone5)
iterativeCone5JetTracksAssociatorAtVertex.jets = 'ZSPJetCorJetIcone5'
iterativeCone5JetTracksAssociatorAtCaloFace.jets = 'ZSPJetCorJetIcone5'
iterativeCone5JetExtender.jets = 'ZSPJetCorJetIcone5'

