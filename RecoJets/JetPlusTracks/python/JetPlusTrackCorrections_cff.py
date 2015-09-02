import FWCore.ParameterSet.Config as cms
# ---------- Add assigned jet-track association

from RecoJets.JetAssociationProducers.ak4JTA_cff import *
ak4JetTracksAssociatorAtVertexJPT = ak4JetTracksAssociatorAtVertex.clone()
ak4JetTracksAssociatorAtVertexJPT.useAssigned = cms.bool(True)
ak4JetTracksAssociatorAtVertexJPT.pvSrc = cms.InputTag("offlinePrimaryVertices")

# ---------- Tight Electron ID

from RecoEgamma.ElectronIdentification.electronIdSequence_cff import eidTight
JPTeidTight = eidTight.clone()


# ---------- Module definition
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *


JetPlusTrackZSPCorJetAntiKt4 = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorAntiKt4),
    src = cms.InputTag("ak4CaloJets"),
    tagName = cms.vstring('ZSP_CMSSW390_Akt_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(False),
    srcPVs = cms.InputTag('offlinePrimaryVertices'),    
    alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt4'),
    ptCUT = cms.double(15.)
    )

JetPlusTrackZSPCorJetAntiKt4.JetTracksAssociationAtVertex = cms.InputTag("ak4JetTracksAssociatorAtVertexJPT")
JetPlusTrackZSPCorJetAntiKt4.JetTracksAssociationAtCaloFace = cms.InputTag("ak4JetTracksAssociatorAtCaloFace")
JetPlusTrackZSPCorJetAntiKt4.JetSplitMerge = cms.int32(2)

### ---------- Sequences

# Anti-Kt

JetPlusTrackCorrectionsAntiKt4 = cms.Sequence(
    JPTeidTight*
    ak4JetTracksAssociatorAtVertexJPT*
    ak4JetTracksAssociatorAtCaloFace*
    JetPlusTrackZSPCorJetAntiKt4
    )

# For backward-compatiblity (but to be deprecated!)
JetPlusTrackCorrections = cms.Sequence(JetPlusTrackCorrectionsAntiKt4)
