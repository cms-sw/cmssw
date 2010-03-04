import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.RecoJetAssociations_cff import *

# ---------- Tight Electron ID

from RecoEgamma.ElectronIdentification.electronIdSequence_cff import eidTight
JPTeidTight = eidTight.clone()


# ---------- Module definition
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *

JetPlusTrackZSPCorJetIcone5 = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("iterativeCone5CaloJets"),    
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(True),
    srcPVs  = cms.InputTag('offlinePrimaryVertices'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetIcone5')
    )
    
JetPlusTrackZSPCorJetIcone5.JetTracksAssociationAtVertex = cms.InputTag("JPTiterativeCone5JetTracksAssociatorAtVertex")
JetPlusTrackZSPCorJetIcone5.JetTracksAssociationAtCaloFace = cms.InputTag("JPTiterativeCone5JetTracksAssociatorAtCaloFace")
JetPlusTrackZSPCorJetIcone5.JetSplitMerge = cms.int32(0)

JetPlusTrackZSPCorJetSiscone5 = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("sisCone5CaloJets"),
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0), 
    UseZSP = cms.bool(True), 
    srcPVs  = cms.InputTag('offlinePrimaryVertices'),  
    alias = cms.untracked.string('JetPlusTrackZSPCorJetSiscone5')
    )

JetPlusTrackZSPCorJetSiscone5.JetTracksAssociationAtVertex = cms.InputTag("JPTSisCone5JetTracksAssociatorAtVertex")
JetPlusTrackZSPCorJetSiscone5.JetTracksAssociationAtCaloFace = cms.InputTag("JPTSisCone5JetTracksAssociatorAtCaloFace")
JetPlusTrackZSPCorJetSiscone5.JetSplitMerge = cms.int32(1)

JetPlusTrackZSPCorJetAntiKt5 = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("ak5CaloJets"),
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(True),
    srcPVs  = cms.InputTag('offlinePrimaryVertices'),    
    alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt5')
    )

JetPlusTrackZSPCorJetAntiKt5.JetTracksAssociationAtVertex = cms.InputTag("JPTAntiKt5JetTracksAssociatorAtVertex")
JetPlusTrackZSPCorJetAntiKt5.JetTracksAssociationAtCaloFace = cms.InputTag("JPTAntiKt5JetTracksAssociatorAtCaloFace")
JetPlusTrackZSPCorJetAntiKt5.JetSplitMerge = cms.int32(2)


##### Association 

# IC
from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import*

JPTiterativeCone5JetTracksAssociatorAtVertex = iterativeCone5JetTracksAssociatorAtVertex.clone() 
JPTiterativeCone5JetTracksAssociatorAtVertex.jets = cms.InputTag("iterativeCone5CaloJets")

JPTiterativeCone5JetTracksAssociatorAtCaloFace = iterativeCone5JetTracksAssociatorAtCaloFace.clone()
JPTiterativeCone5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("iterativeCone5CaloJets")

JPTiterativeCone5JetExtender = iterativeCone5JetExtender.clone() 
JPTiterativeCone5JetExtender.jets = cms.InputTag("iterativeCone5CaloJets")
JPTiterativeCone5JetExtender.jet2TracksAtCALO = cms.InputTag("JPTiterativeCone5JetTracksAssociatorAtCaloFace")
JPTiterativeCone5JetExtender.jet2TracksAtVX = cms.InputTag("JPTiterativeCone5JetTracksAssociatorAtVertex")

# SisCone
from RecoJets.JetAssociationProducers.sisCone5JTA_cff import*

JPTSisCone5JetTracksAssociatorAtVertex = sisCone5JetTracksAssociatorAtVertex.clone()
JPTSisCone5JetTracksAssociatorAtVertex.jets = cms.InputTag("sisCone5CaloJets")

JPTSisCone5JetTracksAssociatorAtCaloFace = sisCone5JetTracksAssociatorAtCaloFace.clone()
JPTSisCone5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("sisCone5CaloJets")

JPTSisCone5JetExtender = sisCone5JetExtender.clone()
JPTSisCone5JetExtender.jets = cms.InputTag("sisCone5CaloJets")
JPTSisCone5JetExtender.jet2TracksAtCALO = cms.InputTag("JPTSisCone5JetTracksAssociatorAtCaloFace")
JPTSisCone5JetExtender.jet2TracksAtVX = cms.InputTag("JPTSisCone5JetTracksAssociatorAtVertex")

# Anti-Kt
from RecoJets.JetAssociationProducers.ak5JTA_cff import*

JPTAntiKt5JetTracksAssociatorAtVertex = ak5JetTracksAssociatorAtVertex.clone()
JPTAntiKt5JetTracksAssociatorAtVertex.jets = cms.InputTag("ak5CaloJets")

JPTAntiKt5JetTracksAssociatorAtCaloFace = ak5JetTracksAssociatorAtCaloFace.clone()
JPTAntiKt5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ak5CaloJets")

JPTAntiKt5JetExtender = ak5JetExtender.clone()
JPTAntiKt5JetExtender.jets = cms.InputTag("ak5CaloJets")
JPTAntiKt5JetExtender.jet2TracksAtCALO = cms.InputTag("JPTAntiKt5JetTracksAssociatorAtCaloFace")
JPTAntiKt5JetExtender.jet2TracksAtVX = cms.InputTag("JPTAntiKt5JetTracksAssociatorAtVertex")

### ---------- Sequences

# IC5

JPTrecoJetAssociationsIcone5 = cms.Sequence(
    JPTeidTight*
    JPTiterativeCone5JetTracksAssociatorAtVertex*
    JPTiterativeCone5JetTracksAssociatorAtCaloFace*
    JPTiterativeCone5JetExtender
    )

JetPlusTrackCorrectionsIcone5 = cms.Sequence(
    JPTrecoJetAssociationsIcone5*
    JetPlusTrackZSPCorJetIcone5
    )

# SC5

JPTrecoJetAssociationsSisCone5 = cms.Sequence(
    JPTeidTight*
    JPTSisCone5JetTracksAssociatorAtVertex*
    JPTSisCone5JetTracksAssociatorAtCaloFace*
    JPTSisCone5JetExtender
    )

JetPlusTrackCorrectionsSisCone5 = cms.Sequence(
    JPTrecoJetAssociationsSisCone5*
    JetPlusTrackZSPCorJetSiscone5
    )

# Anti-Kt

JPTrecoJetAssociationsAntiKt5 = cms.Sequence(
    JPTeidTight*
    JPTAntiKt5JetTracksAssociatorAtVertex*
    JPTAntiKt5JetTracksAssociatorAtCaloFace*
    JPTAntiKt5JetExtender
    )

JetPlusTrackCorrectionsAntiKt5 = cms.Sequence(
    JPTrecoJetAssociationsAntiKt5*
    JetPlusTrackZSPCorJetAntiKt5
    )

# For backward-compatiblity (but to be deprecated!)
JetPlusTrackCorrections = JetPlusTrackCorrectionsIcone5
