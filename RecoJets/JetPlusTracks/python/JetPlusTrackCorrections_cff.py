import FWCore.ParameterSet.Config as cms

# ---------- Tight Electron ID

from RecoEgamma.ElectronIdentification.electronIdSequence_cff import eidTight
JPTeidTight = eidTight.clone()


# ---------- Module definition
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *

JetPlusTrackZSPCorJetIcone5 = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("iterativeCone5CaloJets"),    
#    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
#    tagName = cms.vstring('ZSP_CMSSW361_Akt_05_PU0'),
    tagName = cms.vstring('ZSP_CMSSW390_Akt_05_PU0'),
#    tagName = cms.vstring('ZSP_CMSSW356_Akt_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(False),
    srcPVs  = cms.InputTag('offlinePrimaryVertices'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetIcone5')
    )
    
JetPlusTrackZSPCorJetIcone5.JetSplitMerge = cms.int32(0)

JetPlusTrackZSPCorJetAntiKt5 = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("ak5CaloJets"),
#    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
#    tagName = cms.vstring('ZSP_CMSSW361_Akt_05_PU0'),
    tagName = cms.vstring('ZSP_CMSSW390_Akt_05_PU0'),
#    tagName = cms.vstring('ZSP_CMSSW356_Akt_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(False),
    srcPVs  = cms.InputTag('offlinePrimaryVertices'),    
    alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt5')
    )

JetPlusTrackZSPCorJetAntiKt5.JetTracksAssociationAtVertex = cms.InputTag("ak5JetTracksAssociatorAtVertex")
JetPlusTrackZSPCorJetAntiKt5.JetTracksAssociationAtCaloFace = cms.InputTag("ak5JetTracksAssociatorAtCaloFace")
JetPlusTrackZSPCorJetAntiKt5.JetSplitMerge = cms.int32(2)


JetPlusTrackZSPCorJetSiscone5 = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("ak5CaloJets"),
#    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
#    tagName = cms.vstring('ZSP_CMSSW361_Akt_05_PU0'),
    tagName = cms.vstring('ZSP_CMSSW390_Akt_05_PU0'),
#    tagName = cms.vstring('ZSP_CMSSW356_Akt_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(False),
    srcPVs  = cms.InputTag('offlinePrimaryVertices'),    
    alias = cms.untracked.string('JetPlusTrackZSPCorJetSiscone5')
    )

JetPlusTrackZSPCorJetSiscone5.JetTracksAssociationAtVertex = cms.InputTag("sisCone5JetTracksAssociatorAtVertex")
JetPlusTrackZSPCorJetSiscone5.JetTracksAssociationAtCaloFace = cms.InputTag("sisCone5JetTracksAssociatorAtCaloFace")
JetPlusTrackZSPCorJetSiscone5.JetSplitMerge = cms.int32(1)




### ---------- Sequences

# IC5

JetPlusTrackCorrectionsIcone5 = cms.Sequence(
    JPTeidTight*
    JetPlusTrackZSPCorJetIcone5
    )

# SC5

JetPlusTrackCorrectionsSisCone5 = cms.Sequence(
    JPTeidTight*
    JetPlusTrackZSPCorJetSiscone5
    )

# Anti-Kt

JetPlusTrackCorrectionsAntiKt5 = cms.Sequence(
    JPTeidTight*
    JetPlusTrackZSPCorJetAntiKt5
    )

# For backward-compatiblity (but to be deprecated!)
JetPlusTrackCorrections = cms.Sequence(JetPlusTrackCorrectionsIcone5)
