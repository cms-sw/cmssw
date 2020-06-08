import FWCore.ParameterSet.Config as cms
# ---------- Add assigned jet-track association

from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
trackExtrapolator.trackSrc = cms.InputTag("produceTracks:tracksFromPF")
trackExtrapolator.trackQuality = cms.string('highPurity')

from RecoJets.JetAssociationProducers.ak4JTA_cff import *
ak4JetTracksAssociatorAtVertexJPTPAT = ak4JetTracksAssociatorAtVertex.clone()
ak4JetTracksAssociatorAtVertexJPTPAT.useAssigned = cms.bool(True)
ak4JetTracksAssociatorAtVertexJPTPAT.pvSrc = cms.InputTag("offlineSlimmedPrimaryVertices")
ak4JetTracksAssociatorAtVertexJPTPAT.jets = cms.InputTag("slimmedCaloJets")
ak4JetTracksAssociatorAtVertexJPTPAT.tracks = cms.InputTag("produceTracks:tracksFromPF")
ak4JetTracksAssociatorAtCaloFaceJPTPAT = ak4JetTracksAssociatorAtCaloFace.clone()
ak4JetTracksAssociatorAtCaloFaceJPTPAT.jets = cms.InputTag("slimmedCaloJets")
ak4JetTracksAssociatorAtCaloFaceJPTPAT.tracks = cms.InputTag("produceTracks:tracksFromPF")
ak4JetExtenderJPTPAT = ak4JetExtender.clone()
ak4JetExtenderJPTPAT.jets = cms.InputTag("slimmedCaloJets")
ak4JetExtenderJPTPAT.jet2TracksAtCALO = cms.InputTag("ak4JetTracksAssociatorAtCaloFaceJPTPAT")
ak4JetExtenderJPTPAT.jet2TracksAtVX = cms.InputTag("ak4JetTracksAssociatorAtVertexJPTPAT")

recoJetAssociationsAntiKt4JPTPAT = cms.Sequence(
    trackExtrapolator*
    ak4JetTracksAssociatorAtVertexJPTPAT*
    ak4JetTracksAssociatorAtCaloFaceJPTPAT*
    ak4JetExtenderJPTPAT
)

# ---------- Tight Electron ID
# commented, needed for CMSSW80X and earlier
#from RecoEgamma.ElectronIdentification.electronIdSequence_cff import eidTight
#JPTeidTightPAT = eidTight.clone()
#from EgammaUser.EgammaPostRecoTools.EgammaPostRecoTools import setupEgammaPostRecoSeq
#setupEgammaPostRecoSeq(process,
#                       era='2017-Nov17ReReco')
#
# ---------- Module definition
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *

JetPlusTrackZSPCorJetAntiKt4PAT = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorAntiKt4),
    src = cms.InputTag("slimmedCaloJets"),
    srcTrackJets = cms.InputTag("ak4TrackJets"), 
    srcAddCaloJets = cms.InputTag('JetPlusTrackAddonSeedPAT:CaloJetAddonSeed'),
    extrapolations = cms.InputTag("trackExtrapolator"),
    tagName = cms.vstring('ZSP_CMSSW390_Akt_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(False),
    srcPVs = cms.InputTag('offlineSlimmedPrimaryVertices'),    
    alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt4'),
    ptCUT = cms.double(15.),
    PAT   = cms.bool(True)
    )

JetPlusTrackZSPCorJetAntiKt4PAT.JetTracksAssociationAtVertex = cms.InputTag("ak4JetTracksAssociatorAtVertexJPTPAT")
JetPlusTrackZSPCorJetAntiKt4PAT.JetTracksAssociationAtCaloFace = cms.InputTag("ak4JetTracksAssociatorAtCaloFaceJPTPAT")
JetPlusTrackZSPCorJetAntiKt4PAT.Muons = cms.InputTag("slimmedMuons")
JetPlusTrackZSPCorJetAntiKt4PAT.Electrons = cms.InputTag("slimmedElectrons")
JetPlusTrackZSPCorJetAntiKt4PAT.JetSplitMerge = cms.int32(2)
JetPlusTrackZSPCorJetAntiKt4PAT.UseReco = cms.bool(False)
### ---------- Sequences

#from RecoJets.JetProducers.TracksForJets_cff import *
#from RecoJets.Configuration.RecoTrackJets_cff import *
#from CommonTools.RecoAlgos.TrackWithVertexSelector_cfi import *
#from CommonTools.RecoAlgos.TrackWithVertexSelectorParams_cff import *
#trackWithVertexSelectorParams.vertexTag = cms.InputTag('offlineSlimmedPrimaryVertices')
#trackWithVertexSelectorParams.src = cms.InputTag('produceTracks:tracksFromPF')

# Anti-Kt

JetPlusTrackCorrectionsAntiKt4TaskPAT = cms.Task(
    trackExtrapolator,
    ak4JetTracksAssociatorAtVertexJPTPAT,
    ak4JetTracksAssociatorAtCaloFaceJPTPAT,
    ak4JetExtenderJPTPAT,
    JetPlusTrackZSPCorJetAntiKt4PAT
    )

JetPlusTrackCorrectionsAntiKt4PAT = cms.Sequence(JetPlusTrackCorrectionsAntiKt4TaskPAT)

# For backward-compatiblity (but to be deprecated!)

JetPlusTrackCorrectionsPAT = cms.Sequence(JetPlusTrackCorrectionsAntiKt4PAT)
