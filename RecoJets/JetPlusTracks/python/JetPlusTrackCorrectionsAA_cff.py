import FWCore.ParameterSet.Config as cms


# ---------- Tight Electron ID
from RecoEgamma.ElectronIdentification.electronIdSequence_cff import eidTight
JPTeidTight = eidTight.clone()


# ---------- Module definition
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *
JetPlusTrackZSPCorJetAntiKtPu4 = cms.EDProducer(
    "JetPlusTrackProducerAA",
    cms.PSet(JPTZSPCorrectorAntiKt4),
    src = cms.InputTag("akPu4CaloJets"),
    coneSize = cms.double(0.4),
    tracks = cms.InputTag("hiGoodMergedTracks"),
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0), 
    UseZSP = cms.bool(False),   
    srcPVs  = cms.InputTag('hiSelectedVertex'),
    alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt4'),
    extrapolations = cms.InputTag("trackExtrapolator")
    )

JetPlusTrackZSPCorJetAntiKtPu4.JetTracksAssociationAtVertex = "JPTAntiKtPu4JetTracksAssociatorAtVertex"
JetPlusTrackZSPCorJetAntiKtPu4.JetTracksAssociationAtCaloFace = "JPTAntiKtPu4JetTracksAssociatorAtCaloFace"
JetPlusTrackZSPCorJetAntiKtPu4.JetSplitMerge = 2
JetPlusTrackZSPCorJetAntiKtPu4.UseTrackQuality = False
JetPlusTrackZSPCorJetAntiKtPu4.UseMuons = False
JetPlusTrackZSPCorJetAntiKtPu4.UseElectron = False
JetPlusTrackZSPCorJetAntiKtPu4.EfficiencyMap = "CondFormats/JetMETObjects/data/CMSSW_538HI_TrackNonEff.txt"

from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
JPTtrackExtrapolatorAA = trackExtrapolator.clone(
                         trackSrc = "hiGoodMergedTracks",
                         trackQuality = 'highPurity'
)

from RecoJets.JetAssociationProducers.ak4JTA_cff import *
JPTAntiKtPu4JetTracksAssociatorAtVertex = ak4JetTracksAssociatorAtVertex.clone(
                                          useAssigned = True,
                                          pvSrc = "hiSelectedVertex",
                                          jets = "akPu4CaloJets",
                                          tracks = "hiGoodMergedTracks"
)
JPTAntiKtPu4JetTracksAssociatorAtCaloFace = ak4JetTracksAssociatorAtCaloFace.clone(
                                            jets = "akPu4CaloJets",
                                            tracks = "hiGoodMergedTracks",
                                            extrapolations = "JPTtrackExtrapolatorAA"
)
ak4JetExtenderJPTPAT = ak4JetExtender.clone(
                       jets = "akPu4CaloJets",
                       jet2TracksAtCALO = "JPTAntiKtPu4JetTracksAssociatorAtCaloFace",
                       jet2TracksAtVX = "JPTAntiKtPu4JetTracksAssociatorAtVertex"
)

# Task definition
JetPlusTrackCorrectionsAntiKtPu4Task = cms.Task(
    JPTtrackExtrapolatorAA,
    JPTAntiKtPu4JetTracksAssociatorAtVertex,
    JPTAntiKtPu4JetTracksAssociatorAtCaloFace*
    JPTAntiKtPu4JetExtender,
    JetPlusTrackZSPCorJetAntiKtPu4
    )
