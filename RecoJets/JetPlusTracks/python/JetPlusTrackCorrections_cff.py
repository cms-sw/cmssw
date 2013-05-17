import FWCore.ParameterSet.Config as cms
# ---------- Add assigned jet-track association

from RecoJets.JetAssociationProducers.ak5JTA_cff import *
ak5JetTracksAssociatorAtVertexJPT = ak5JetTracksAssociatorAtVertex.clone()
ak5JetTracksAssociatorAtVertexJPT.useAssigned = cms.bool(True)
ak5JetTracksAssociatorAtVertexJPT.pvSrc = cms.InputTag("offlinePrimaryVertices")

from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import *
iterativeCone5JetTracksAssociatorAtVertexJPT = iterativeCone5JetTracksAssociatorAtVertex.clone()
iterativeCone5JetTracksAssociatorAtVertexJPT.useAssigned = cms.bool(True)
iterativeCone5JetTracksAssociatorAtVertexJPT.pvSrc = cms.InputTag("offlinePrimaryVertices")

from RecoJets.JetAssociationProducers.sisCone5JTA_cff import *
sisCone5JetTracksAssociatorAtVertexJPT = sisCone5JetTracksAssociatorAtVertex.clone()
sisCone5JetTracksAssociatorAtVertexJPT.useAssigned = cms.bool(True)
sisCone5JetTracksAssociatorAtVertexJPT.pvSrc = cms.InputTag("offlinePrimaryVertices")


# ---------- Tight Electron ID

from RecoEgamma.ElectronIdentification.electronIdSequence_cff import eidTight
JPTeidTight = eidTight.clone()


# ---------- Module definition
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *

JetPlusTrackZSPCorJetIcone5 = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("iterativeCone5CaloJets"),    
    tagName = cms.vstring('ZSP_CMSSW390_Akt_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(False),
    srcPVs  = cms.InputTag('offlinePrimaryVertices'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetIcone5'),
    ptCUT = cms.double(15.)
    )
    
JetPlusTrackZSPCorJetIcone5.JetSplitMerge = cms.int32(0)
JetPlusTrackZSPCorJetIcone5.JetTracksAssociationAtVertex = cms.InputTag("iterativeCone5JetTracksAssociatorAtVertexJPT") 


JetPlusTrackZSPCorJetAntiKt5 = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("ak5CaloJets"),
    tagName = cms.vstring('ZSP_CMSSW390_Akt_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(False),
    srcPVs = cms.InputTag('offlinePrimaryVertices'),    
    alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt5'),
    ptCUT = cms.double(15.)
    )

JetPlusTrackZSPCorJetAntiKt5.JetTracksAssociationAtVertex = cms.InputTag("ak5JetTracksAssociatorAtVertexJPT")
JetPlusTrackZSPCorJetAntiKt5.JetTracksAssociationAtCaloFace = cms.InputTag("ak5JetTracksAssociatorAtCaloFace")
JetPlusTrackZSPCorJetAntiKt5.JetSplitMerge = cms.int32(2)


JetPlusTrackZSPCorJetSiscone5 = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("ak5CaloJets"),
    tagName = cms.vstring('ZSP_CMSSW390_Akt_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(False),
    srcPVs  = cms.InputTag('offlinePrimaryVertices'),    
    alias = cms.untracked.string('JetPlusTrackZSPCorJetSiscone5'),
    ptCUT = cms.double(15.)
    )

JetPlusTrackZSPCorJetSiscone5.JetTracksAssociationAtVertex = cms.InputTag("sisCone5JetTracksAssociatorAtVertexJPT")
JetPlusTrackZSPCorJetSiscone5.JetTracksAssociationAtCaloFace = cms.InputTag("sisCone5JetTracksAssociatorAtCaloFace")
JetPlusTrackZSPCorJetSiscone5.JetSplitMerge = cms.int32(1)




### ---------- Sequences

# IC5

JetPlusTrackCorrectionsIcone5 = cms.Sequence(
    JPTeidTight*
    iterativeCone5JetTracksAssociatorAtVertexJPT*
    JetPlusTrackZSPCorJetIcone5
    )

# SC5

JetPlusTrackCorrectionsSisCone5 = cms.Sequence(
    JPTeidTight*
    sisCone5JetTracksAssociatorAtVertexJPT*
    JetPlusTrackZSPCorJetSiscone5
    )

# Anti-Kt

JetPlusTrackCorrectionsAntiKt5 = cms.Sequence(
    JPTeidTight*
    ak5JetTracksAssociatorAtVertexJPT*
    JetPlusTrackZSPCorJetAntiKt5
    )

# For backward-compatiblity (but to be deprecated!)
JetPlusTrackCorrections = cms.Sequence(JetPlusTrackCorrectionsIcone5)
