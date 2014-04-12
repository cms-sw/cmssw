import FWCore.ParameterSet.Config as cms

# keep IC5 association for backward compatibility for external use
from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import *
from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
#JPTtrackExtrapolatorAA = trackExtrapolator.clone()
trackExtrapolator.trackSrc = cms.InputTag("hiGoodMergedTracks")

# standard associations
from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import *
from RecoJets.JetAssociationProducers.sisCone5JTA_cff import *
from RecoJets.JetAssociationProducers.kt4JTA_cff import *
from RecoJets.JetAssociationProducers.ak5JTA_cff import *
from RecoJets.JetAssociationProducers.ak7JTA_cff import *

# ---------- Tight Electron ID

from RecoEgamma.ElectronIdentification.electronIdSequence_cff import eidTight
JPTeidTight = eidTight.clone()


# ---------- Module definition
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *

JetPlusTrackZSPCorJetIconePu5 = cms.EDProducer(
    "JetPlusTrackProducerAA",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("iterativeConePu5CaloJets"),
    coneSize = cms.double(0.5),
    tracks = cms.InputTag("hiGoodMergedTracks"),    
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(False),
    srcPVs  = cms.InputTag('hiSelectedVertex'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetIcone5'),
#=>
    extrapolations = cms.InputTag("trackExtrapolator")
    )
    
JetPlusTrackZSPCorJetIconePu5.JetTracksAssociationAtVertex = cms.InputTag("JPTiterativeConePu5JetTracksAssociatorAtVertex")
JetPlusTrackZSPCorJetIconePu5.JetTracksAssociationAtCaloFace = cms.InputTag("JPTiterativeConePu5JetTracksAssociatorAtCaloFace")
JetPlusTrackZSPCorJetIconePu5.JetSplitMerge = cms.int32(0)
JetPlusTrackZSPCorJetIconePu5.UseTrackQuality = cms.bool(False)
JetPlusTrackZSPCorJetIconePu5.UseMuons = cms.bool(False)
JetPlusTrackZSPCorJetIconePu5.UseElectrons = cms.bool(False)
JetPlusTrackZSPCorJetIconePu5.EfficiencyMap = cms.string("CondFormats/JetMETObjects/data/CMSSW_538HI_TrackNonEff.txt")
JetPlusTrackZSPCorJetIconePu5.UseOutOfVertexTracks = cms.bool(False)

JetPlusTrackZSPCorJetSisconePu5 = cms.EDProducer(
    "JetPlusTrackProducerAA",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("sisConePu5CaloJets"),
    coneSize = cms.double(0.5),
    tracks = cms.InputTag("hiGoodMergedTracks"),
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0), 
    UseZSP = cms.bool(False),   
    srcPVs  = cms.InputTag('hiSelectedVertex'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetSiscone5'),
#=>
    extrapolations = cms.InputTag("trackExtrapolator")   
    )

JetPlusTrackZSPCorJetSisconePu5.JetTracksAssociationAtVertex = cms.InputTag("JPTSisConePu5JetTracksAssociatorAtVertex")
JetPlusTrackZSPCorJetSisconePu5.JetTracksAssociationAtCaloFace = cms.InputTag("JPTSisConePu5JetTracksAssociatorAtCaloFace")
JetPlusTrackZSPCorJetSisconePu5.JetSplitMerge = cms.int32(1)
JetPlusTrackZSPCorJetSisconePu5.UseTrackQuality = cms.bool(False)
JetPlusTrackZSPCorJetSisconePu5.UseMuons = cms.bool(False)
JetPlusTrackZSPCorJetSisconePu5.UseElectrons = cms.bool(False)
JetPlusTrackZSPCorJetSisconePu5.EfficiencyMap = cms.string("CondFormats/JetMETObjects/data/CMSSW_538HI_TrackNonEff.txt")
JetPlusTrackZSPCorJetSisconePu5.UseOutOfVertexTracks = cms.bool(False)

JetPlusTrackZSPCorJetAntiKtPu5 = cms.EDProducer(
    "JetPlusTrackProducerAA",
    cms.PSet(JPTZSPCorrectorICone5),
    src = cms.InputTag("akPu5CaloJets"),
    coneSize = cms.double(0.5),
    tracks = cms.InputTag("hiGoodMergedTracks"),
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0), 
    UseZSP = cms.bool(False),   
    srcPVs  = cms.InputTag('hiSelectedVertex'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt5'),
#=>
    extrapolations = cms.InputTag("trackExtrapolator")
    )

JetPlusTrackZSPCorJetAntiKtPu5.JetTracksAssociationAtVertex = cms.InputTag("JPTAntiKtPu5JetTracksAssociatorAtVertex")
JetPlusTrackZSPCorJetAntiKtPu5.JetTracksAssociationAtCaloFace = cms.InputTag("JPTAntiKtPu5JetTracksAssociatorAtCaloFace")

JetPlusTrackZSPCorJetAntiKtPu5.JetSplitMerge = cms.int32(2)
JetPlusTrackZSPCorJetAntiKtPu5.UseTrackQuality = cms.bool(False)
JetPlusTrackZSPCorJetAntiKtPu5.UseMuons = cms.bool(False)
JetPlusTrackZSPCorJetAntiKtPu5.UseElectron = cms.bool(False)
JetPlusTrackZSPCorJetAntiKtPu5.EfficiencyMap = cms.string("CondFormats/JetMETObjects/data/CMSSW_538HI_TrackNonEff.txt")

##### Association 

#===> TRACK
JPTtrackExtrapolatorAA = trackExtrapolator.clone()
JPTtrackExtrapolatorAA.trackSrc = cms.InputTag("hiGoodMergedTracks")
JPTtrackExtrapolatorAA.trackQuality = cms.string('highPurity')
##JPTtrackExtrapolatorAA.trackQuality = cms.string('loose')
#===>

# IC
from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import*

JPTiterativeConePu5JetTracksAssociatorAtVertex = iterativeCone5JetTracksAssociatorAtVertex.clone() 
JPTiterativeConePu5JetTracksAssociatorAtVertex.jets = cms.InputTag("iterativeConePu5CaloJets")
JPTiterativeConePu5JetTracksAssociatorAtVertex.tracks = cms.InputTag("hiGoodMergedTracks")

JPTiterativeConePu5JetTracksAssociatorAtCaloFace = iterativeCone5JetTracksAssociatorAtCaloFace.clone()
JPTiterativeConePu5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("iterativeConePu5CaloJets")
JPTiterativeConePu5JetTracksAssociatorAtCaloFace.tracks = cms.InputTag("hiGoodMergedTracks")

#===>
#JPTiterativeConePu5JetTracksAssociatorAtCaloFace.extrapolations = cms.InputTag("JPTtrackExtrapolatorAA")
#JPTiterativeConePu5JetTracksAssociatorAtCaloFace.trackQuality = cms.string('highPurity')
#===>

JPTiterativeConePu5JetExtender = iterativeCone5JetExtender.clone() 
JPTiterativeConePu5JetExtender.jets = cms.InputTag("iterativeConePu5CaloJets")
JPTiterativeConePu5JetExtender.jet2TracksAtCALO = cms.InputTag("JPTiterativeConePu5JetTracksAssociatorAtCaloFace")
JPTiterativeConePu5JetExtender.jet2TracksAtVX = cms.InputTag("JPTiterativeConePu5JetTracksAssociatorAtVertex")

# SisCone
from RecoJets.JetAssociationProducers.sisCone5JTA_cff import*

JPTSisConePu5JetTracksAssociatorAtVertex = sisCone5JetTracksAssociatorAtVertex.clone()
JPTSisConePu5JetTracksAssociatorAtVertex.jets = cms.InputTag("sisConePu5CaloJets")
JPTSisConePu5JetTracksAssociatorAtVertex.tracks = cms.InputTag("hiGoodMergedTracks")

JPTSisConePu5JetTracksAssociatorAtCaloFace = sisCone5JetTracksAssociatorAtCaloFace.clone()
JPTSisConePu5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("sisConePu5CaloJets")
JPTSisConePu5JetTracksAssociatorAtCaloFace.tracks = cms.InputTag("hiGoodMergedTracks")

JPTSisConePu5JetExtender = sisCone5JetExtender.clone()
JPTSisConePu5JetExtender.jets = cms.InputTag("sisConePu5CaloJets")
JPTSisConePu5JetExtender.jet2TracksAtCALO = cms.InputTag("JPTSisConePu5JetTracksAssociatorAtCaloFace")
JPTSisConePu5JetExtender.jet2TracksAtVX = cms.InputTag("JPTSisConePu5JetTracksAssociatorAtVertex")

# Anti-Kt
from RecoJets.JetAssociationProducers.ak5JTA_cff import*

JPTAntiKtPu5JetTracksAssociatorAtVertex = ak5JetTracksAssociatorAtVertex.clone()
JPTAntiKtPu5JetTracksAssociatorAtVertex.jets = cms.InputTag("akPu5CaloJets")
JPTAntiKtPu5JetTracksAssociatorAtVertex.tracks = cms.InputTag("hiGoodMergedTracks")

JPTAntiKtPu5JetTracksAssociatorAtCaloFace = ak5JetTracksAssociatorAtCaloFace.clone()
JPTAntiKtPu5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("akPu5CaloJets")
JPTAntiKtPu5JetTracksAssociatorAtCaloFace.tracks = cms.InputTag("hiGoodMergedTracks")

JPTAntiKtPu5JetExtender = ak5JetExtender.clone()
JPTAntiKtPu5JetExtender.jets = cms.InputTag("akPu5CaloJets")
JPTAntiKtPu5JetExtender.jet2TracksAtCALO = cms.InputTag("JPTAntiKtPu5JetTracksAssociatorAtCaloFace")
JPTAntiKtPu5JetExtender.jet2TracksAtVX = cms.InputTag("JPTAntiKtPu5JetTracksAssociatorAtVertex")

### ---------- Sequences

# IC5

JPTrecoJetAssociationsIconePu5 = cms.Sequence(
    trackExtrapolator*
###    JPTtrackExtrapolatorAA*
    JPTiterativeConePu5JetTracksAssociatorAtVertex*
    JPTiterativeConePu5JetTracksAssociatorAtCaloFace*
    JPTiterativeConePu5JetExtender
    )

JetPlusTrackCorrectionsIconePu5 = cms.Sequence(
    JPTrecoJetAssociationsIconePu5*
    JetPlusTrackZSPCorJetIconePu5
    )

# SC5

JPTrecoJetAssociationsSisConePu5 = cms.Sequence(
    trackExtrapolator*
    JPTSisConePu5JetTracksAssociatorAtVertex*
    JPTSisConePu5JetTracksAssociatorAtCaloFace*
    JPTSisConePu5JetExtender
    )

JetPlusTrackCorrectionsSisConePu5 = cms.Sequence(
    JPTrecoJetAssociationsSisConePu5*
    JetPlusTrackZSPCorJetSisconePu5
    )

# Anti-Kt

JPTrecoJetAssociationsAntiKtPu5 = cms.Sequence(
    trackExtrapolator*
    JPTAntiKtPu5JetTracksAssociatorAtVertex*
    JPTAntiKtPu5JetTracksAssociatorAtCaloFace*
    JPTAntiKtPu5JetExtender
    )

JetPlusTrackCorrectionsAntiKtPu5 = cms.Sequence(
    JPTrecoJetAssociationsAntiKtPu5*
    JetPlusTrackZSPCorJetAntiKtPu5
    )
