import FWCore.ParameterSet.Config as cms

# File: JetCorrections.cff
# Author: O. Kodolova
# Date: 1/24/07
#
# Jet corrections with tracks for the icone5 jets with ZSP corrections.
# 
from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

from JetMETCorrections.Configuration.JetPlusTrackCorrections_cfi import *

JetPlusTrackZSPCorrectorIcone5 = cms.ESSource(
    "JetPlusTrackCorrectionService",
    JPTZSPCorrectorICone5,
    label = cms.string('JetPlusTrackZSPCorrectorIcone5'),
    )

JetPlusTrackZSPCorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("ZSPJetCorJetIcone5"),
    correctors = cms.vstring('JetPlusTrackZSPCorrectorIcone5'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetIcone5')
)

from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import*

ZSPiterativeCone5JetTracksAssociatorAtVertex = iterativeCone5JetTracksAssociatorAtVertex.clone() 
ZSPiterativeCone5JetTracksAssociatorAtVertex.jets = cms.InputTag("ZSPJetCorJetIcone5")

ZSPiterativeCone5JetTracksAssociatorAtCaloFace = iterativeCone5JetTracksAssociatorAtCaloFace.clone()
ZSPiterativeCone5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ZSPJetCorJetIcone5")

ZSPiterativeCone5JetExtender = iterativeCone5JetExtender.clone() 
ZSPiterativeCone5JetExtender.jets = cms.InputTag("ZSPJetCorJetIcone5")
ZSPiterativeCone5JetExtender.jet2TracksAtCALO = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtCaloFace")
ZSPiterativeCone5JetExtender.jet2TracksAtVX = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtVertex")


ZSPrecoJetAssociations = cms.Sequence(ZSPiterativeCone5JetTracksAssociatorAtVertex*ZSPiterativeCone5JetTracksAssociatorAtCaloFace*ZSPiterativeCone5JetExtender)

JetPlusTrackCorrections = cms.Sequence(ZSPrecoJetAssociations*JetPlusTrackZSPCorJetIcone5)

