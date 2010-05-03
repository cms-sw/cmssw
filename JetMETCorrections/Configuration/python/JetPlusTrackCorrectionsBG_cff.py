import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.RecoJetAssociations_cff import *

# ---------- Service definition 

from JetMETCorrections.Configuration.JetPlusTrackCorrectionsBG_cfi import *

JetPlusTrackZSPCorrectorIcone5BG = cms.ESSource(
    "JetPlusTrackCorrectionServiceAA",
    cms.PSet(JPTZSPCorrectorICone5BG),
    coneSize = cms.double(0.5),
    jets = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
    label = cms.string('JetPlusTrackZSPCorrectorIcone5BG'),
    )

JetPlusTrackZSPCorrectorSiscone5BG = cms.ESSource(
    "JetPlusTrackCorrectionServiceAA",
    cms.PSet(JPTZSPCorrectorICone5BG),
    coneSize = cms.double(0.5),
    jets = cms.InputTag("JetPlusTrackZSPCorJetSiscone5"),
    label = cms.string('JetPlusTrackZSPCorrectorSiscone5BG'),
    )

JetPlusTrackZSPCorrectorAntiKt5BG = cms.ESSource(
    "JetPlusTrackCorrectionServiceAA",
    cms.PSet(JPTZSPCorrectorICone5BG),
    coneSize = cms.double(0.5),
    jets = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    label = cms.string('JetPlusTrackZSPCorrectorAntiKt5BG'),
    )

# ---------- Module definition

JetPlusTrackZSPCorJetIcone5BG = cms.EDProducer(
    "CaloJetCorrectionProducer",
    src = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
    correctors = cms.vstring('JetPlusTrackZSPCorrectorIcone5BG'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetIcone5BG')
    )

JetPlusTrackZSPCorJetSiscone5BG = cms.EDProducer(
    "CaloJetCorrectionProducer",
    src = cms.InputTag("JetPlusTrackZSPCorJetSiscone5"),
    correctors = cms.vstring('JetPlusTrackZSPCorrectorSiscone5BG'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetSiscone5BG')
    )

JetPlusTrackZSPCorJetAntiKt5BG = cms.EDProducer(
    "CaloJetCorrectionProducer",
    src = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    correctors = cms.vstring('JetPlusTrackZSPCorrectorAntiKt5BG'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt5BG')
    )


# IC05
from JetMETCorrections.Configuration.JetPlusTrackCorrections_cff import *
ZSPiterativeCone5JetTracksAssociatorAtVertex.tracks = cms.InputTag("hiSelectedTracks")
ZSPiterativeCone5JetTracksAssociatorAtCaloFace.tracks = cms.InputTag("hiSelectedTracks")

JetPlusTrackZSPCorrectorIcone5.UseMuons = cms.bool(False)
JetPlusTrackZSPCorrectorIcone5.UseElectrons = cms.bool(False)
JetPlusTrackZSPCorrectorIcone5.UseTrackQuality = cms.bool(False)

ZSPrecoJetAssociationsIcone5PU = cms.Sequence(
    ZSPiterativeCone5JetTracksAssociatorAtVertex*
    ZSPiterativeCone5JetTracksAssociatorAtCaloFace*
    ZSPiterativeCone5JetExtender
    )

JetPlusTrackCorrectionsIcone5BG = cms.Sequence(
    ZSPrecoJetAssociationsIcone5PU*
    JetPlusTrackZSPCorJetIcone5*
    JetPlusTrackZSPCorJetIcone5BG
    )



# Siscone
ZSPSisCone5JetTracksAssociatorAtVertex.tracks = cms.InputTag("hiSelectedTracks")
ZSPSisCone5JetTracksAssociatorAtCaloFace.tracks = cms.InputTag("hiSelectedTracks")
JetPlusTrackZSPCorrectorSiscone5.UseMuons = cms.bool(False)
JetPlusTrackZSPCorrectorSiscone5.UseElectrons = cms.bool(False)
JetPlusTrackZSPCorrectorSiscone5.UseTrackQuality = cms.bool(False)

ZSPrecoJetAssociationsSisCone5PU = cms.Sequence(
    ZSPSisCone5JetTracksAssociatorAtVertex*
    ZSPSisCone5JetTracksAssociatorAtCaloFace*
    ZSPSisCone5JetExtender
    )


JetPlusTrackCorrectionsSisCone5BG = cms.Sequence(
    ZSPrecoJetAssociationsSisCone5PU*
    JetPlusTrackZSPCorJetSiscone5*
    JetPlusTrackZSPCorJetSiscone5BG
    )

# Anti-Kt
ZSPAntiKt5JetTracksAssociatorAtVertex.tracks = cms.InputTag("hiSelectedTracks")
ZSPAntiKt5JetTracksAssociatorAtCaloFace.tracks = cms.InputTag("hiSelectedTracks")
JetPlusTrackZSPCorrectorAntiKt5.UseMuons = cms.bool(False)
JetPlusTrackZSPCorrectorAntiKt5.UseElectrons = cms.bool(False)
JetPlusTrackZSPCorrectorAntiKt5.UseTrackQuality = cms.bool(False)
    
ZSPrecoJetAssociationsAntiKt5PU = cms.Sequence(
    ZSPAntiKt5JetTracksAssociatorAtVertex*
    ZSPAntiKt5JetTracksAssociatorAtCaloFace*
    ZSPAntiKt5JetExtender
    )

JetPlusTrackCorrectionsAntiKt5BG = cms.Sequence(
    ZSPSisCone5JetTracksAssociatorAtVertex*
    ZSPSisCone5JetTracksAssociatorAtCaloFace*
    JetPlusTrackZSPCorJetAntiKt5BG
    )
