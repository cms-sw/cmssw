import FWCore.ParameterSet.Config as cms

# File: JetCorrections.cff
# Author: O. Kodolova
# Date: 1/24/07
#
# Jet corrections with tracks for the icone5 jets with ZSP corrections.
# 
from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

from JetMETCorrections.Configuration.JetPlusTrackCorrections_mod_cff import *

# Tight Electron ID
from RecoEgamma.ElectronIdentification.electronIdSequence_cff import eidTight
JPTeidTight = eidTight.clone()
# Service definition 
JetPlusTrackZSPCorrectorIcone5 = cms.ESSource(
    "JetPlusTrackCorrectionService",
    JPTZSPCorrectorICone5,
    label = cms.string('JetPlusTrackZSPCorrectorIcone5'),
    )
JetPlusTrackZSPCorrectorSiscone5 = cms.ESSource(
    "JetPlusTrackCorrectionService",
    JPTZSPCorrectorSisCone5,
    label = cms.string('JetPlusTrackZSPCorrectorSiscone5'),
    )
JetPlusTrackZSPCorrectorAntiKt5 = cms.ESSource(
    "JetPlusTrackCorrectionService",
    JPTZSPCorrectorAntiKt5,
    label = cms.string('JetPlusTrackZSPCorrectorAntiKt5'),
    )

# Module definition
JetPlusTrackZSPCorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("ZSPJetCorJetIcone5"),
    correctors = cms.vstring('JetPlusTrackZSPCorrectorIcone5'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetIcone5')
)
JetPlusTrackZSPCorJetSiscone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("ZSPJetCorJetSiscone5"),
    correctors = cms.vstring('JetPlusTrackZSPCorrectorSiscone5'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetSiscone5')
)
JetPlusTrackZSPCorJetAntiKt5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("ZSPJetCorJetAntiKt5"),
    correctors = cms.vstring('JetPlusTrackZSPCorrectorAntiKt5'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt5')
)

# Jet-Track association
# IC
from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import*

ZSPiterativeCone5JetTracksAssociatorAtVertex = iterativeCone5JetTracksAssociatorAtVertex.clone() 
ZSPiterativeCone5JetTracksAssociatorAtVertex.jets = cms.InputTag("ZSPJetCorJetIcone5")

ZSPiterativeCone5JetTracksAssociatorAtCaloFace = iterativeCone5JetTracksAssociatorAtCaloFace.clone()
ZSPiterativeCone5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ZSPJetCorJetIcone5")

ZSPiterativeCone5JetExtender = iterativeCone5JetExtender.clone() 
ZSPiterativeCone5JetExtender.jets = cms.InputTag("ZSPJetCorJetIcone5")
ZSPiterativeCone5JetExtender.jet2TracksAtCALO = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtCaloFace")
ZSPiterativeCone5JetExtender.jet2TracksAtVX = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtVertex")

# SisCone
from RecoJets.JetAssociationProducers.sisCone5JTA_cff import*

ZSPSisCone5JetTracksAssociatorAtVertex = sisCone5JetTracksAssociatorAtVertex.clone()
ZSPSisCone5JetTracksAssociatorAtVertex.jets = cms.InputTag("ZSPJetCorJetSiscone5")

ZSPSisCone5JetTracksAssociatorAtCaloFace = sisCone5JetTracksAssociatorAtCaloFace.clone()
ZSPSisCone5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ZSPJetCorJetSiscone5")

ZSPSisCone5JetExtender = sisCone5JetExtender.clone()
ZSPSisCone5JetExtender.jets = cms.InputTag("ZSPJetCorJetSiscone5")
ZSPSisCone5JetExtender.jet2TracksAtCALO = cms.InputTag("ZSPSisCone5JetTracksAssociatorAtCaloFace")
ZSPSisCone5JetExtender.jet2TracksAtVX = cms.InputTag("ZSPSisCone5JetTracksAssociatorAtVertex")

# Anti-Kt
from RecoJets.JetAssociationProducers.ak5JTA_cff import*

ZSPAntiKt5JetTracksAssociatorAtVertex = ak5JetTracksAssociatorAtVertex.clone()
ZSPAntiKt5JetTracksAssociatorAtVertex.jets = cms.InputTag("ZSPJetCorJetAntiKt5")

ZSPAntiKt5JetTracksAssociatorAtCaloFace = ak5JetTracksAssociatorAtCaloFace.clone()
ZSPAntiKt5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ZSPJetCorJetAntiKt5")

ZSPAntiKt5JetExtender = ak5JetExtender.clone()
ZSPAntiKt5JetExtender.jets = cms.InputTag("ZSPJetCorJetAntiKt5")
ZSPAntiKt5JetExtender.jet2TracksAtCALO = cms.InputTag("ZSPAntiKt5JetTracksAssociatorAtCaloFace")
ZSPAntiKt5JetExtender.jet2TracksAtVX = cms.InputTag("ZSPAntiKt5JetTracksAssociatorAtVertex")

### Sequences
ZSPrecoJetAssociationsIcone5 = cms.Sequence(JPTeidTight*ZSPiterativeCone5JetTracksAssociatorAtVertex*ZSPiterativeCone5JetTracksAssociatorAtCaloFace*ZSPiterativeCone5JetExtender)
JetPlusTrackCorrectionsIcone5 = cms.Sequence(ZSPrecoJetAssociationsIcone5*JetPlusTrackZSPCorJetIcone5)
ZSPrecoJetAssociationsSisCone5 = cms.Sequence(JPTeidTight*ZSPSisCone5JetTracksAssociatorAtVertex*ZSPSisCone5JetTracksAssociatorAtCaloFace*ZSPSisCone5JetExtender)
JetPlusTrackCorrectionsSisCone5 = cms.Sequence(ZSPrecoJetAssociationsSisCone5*JetPlusTrackZSPCorJetSiscone5)
ZSPrecoJetAssociationsAntiKt5 = cms.Sequence(JPTeidTight*ZSPAntiKt5JetTracksAssociatorAtVertex*ZSPAntiKt5JetTracksAssociatorAtCaloFace*ZSPAntiKt5JetExtender)
JetPlusTrackCorrectionsAntiKt5 = cms.Sequence(ZSPrecoJetAssociationsAntiKt5*JetPlusTrackZSPCorJetAntiKt5)
