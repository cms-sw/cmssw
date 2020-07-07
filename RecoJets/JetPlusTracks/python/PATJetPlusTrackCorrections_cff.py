import FWCore.ParameterSet.Config as cms
# ---------- Add assigned jet-track association

from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
trackExtrapolatorPAT = trackExtrapolator.clone(
                      trackSrc = "produceTracks",
                      trackQuality = 'highPurity'
)

from RecoJets.JetAssociationProducers.ak4JTA_cff import *
ak4JetTracksAssociatorAtVertexJPTPAT = ak4JetTracksAssociatorAtVertex.clone(
                                       useAssigned = True,
                                       pvSrc = "offlineSlimmedPrimaryVertices",
                                       jets = "slimmedCaloJets",
                                       tracks = "produceTracks"
)
ak4JetTracksAssociatorAtCaloFaceJPTPAT = ak4JetTracksAssociatorAtCaloFace.clone(
                                         jets = "slimmedCaloJets",
                                         tracks = "produceTracks",
                                         extrapolations = "trackExtrapolatorPAT"
)
ak4JetExtenderJPTPAT = ak4JetExtender.clone(
                       jets = "slimmedCaloJets",
                       jet2TracksAtCALO = "ak4JetTracksAssociatorAtCaloFaceJPTPAT",
                       jet2TracksAtVX = "ak4JetTracksAssociatorAtVertexJPTPAT"
)

# ---------- Supported Modules

from CommonTools.RecoAlgos.trackFromPackedCandidateProducer_cfi import *
produceTracks = trackFromPackedCandidateProducer.clone()


from RecoJets.JetPlusTracks.jetPlusTrackAddonSeedProducer_cfi import *
JetPlusTrackAddonSeedRecoPAT = jetPlusTrackAddonSeedProducer.clone(
    srcCaloJets = cms.InputTag("slimmedCaloJets"),
    srcPVs = cms.InputTag('offlineSlimmedPrimaryVertices'),
    UsePAT = True
)


from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import *
trackWithVertexRefSelector.vertexTag = cms.InputTag('offlineSlimmedPrimaryVertices')
trackWithVertexRefSelector.src = cms.InputTag('produceTracks')
from RecoJets.JetProducers.TracksForJets_cff import *
from RecoJets.Configuration.RecoTrackJets_cff import *
ak4TrackJets.srcPVs = cms.InputTag('offlineSlimmedPrimaryVertices')
ak4TrackJets.UseOnlyOnePV = cms.bool(True)

# ---------- Module definition

from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *

JetPlusTrackZSPCorJetAntiKt4PAT = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorAntiKt4),
    src = cms.InputTag("slimmedCaloJets"),
    srcTrackJets = cms.InputTag("ak4TrackJets"), 
    srcAddCaloJets = cms.InputTag('JetPlusTrackAddonSeedRecoPAT'),
    extrapolations = cms.InputTag("trackExtrapolatorPAT"),
    tagName = cms.vstring('ZSP_CMSSW390_Akt_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    UseZSP = cms.bool(False),
    srcPVs = cms.InputTag('offlineSlimmedPrimaryVertices'),    
    alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt4'),
    ptCUT = cms.double(15.),
    dRcone = cms.double(0.4)
    )

JetPlusTrackZSPCorJetAntiKt4PAT.JetTracksAssociationAtVertex = "ak4JetTracksAssociatorAtVertexJPTPAT"
JetPlusTrackZSPCorJetAntiKt4PAT.JetTracksAssociationAtCaloFace = "ak4JetTracksAssociatorAtCaloFaceJPTPAT"
JetPlusTrackZSPCorJetAntiKt4PAT.Muons = "slimmedMuons"
JetPlusTrackZSPCorJetAntiKt4PAT.Electrons = "slimmedElectrons"
JetPlusTrackZSPCorJetAntiKt4PAT.JetSplitMerge = 2
JetPlusTrackZSPCorJetAntiKt4PAT.UsePAT = True
### ---------- Sequences

# Anti-Kt

PATJetPlusTrackCorrectionsAntiKt4 = cms.Sequence(
    produceTracks*
    trackWithVertexRefSelector*
    trackRefsForJets*
    ak4TrackJets*
    JetPlusTrackAddonSeedRecoPAT*
    trackExtrapolatorPAT*
    ak4JetTracksAssociatorAtVertexJPTPAT*
    ak4JetTracksAssociatorAtCaloFaceJPTPAT*
    ak4JetExtenderJPTPAT*
    JetPlusTrackZSPCorJetAntiKt4PAT
    )
