import FWCore.ParameterSet.Config as cms
# ---------- Add assigned jet-track association

from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
trackExtrapolatorJPTPAT = trackExtrapolator.clone(
                      trackSrc = "trackFromPackedCandidate",
                      trackQuality = 'highPurity'
)

from RecoJets.JetAssociationProducers.ak4JTA_cff import *
ak4JetTracksAssociatorAtVertexJPTPAT = ak4JetTracksAssociatorAtVertex.clone(
                                       useAssigned = True,
                                       pvSrc = "offlineSlimmedPrimaryVertices",
                                       jets = "slimmedCaloJets",
                                       tracks = "trackFromPackedCandidate"
)
ak4JetTracksAssociatorAtCaloFaceJPTPAT = ak4JetTracksAssociatorAtCaloFace.clone(
                                         jets = "slimmedCaloJets",
                                         tracks = "trackFromPackedCandidate",
                                         extrapolations = "trackExtrapolatorJPTPAT"
)
ak4JetExtenderJPTPAT = ak4JetExtender.clone(
                       jets = "slimmedCaloJets",
                       jet2TracksAtCALO = "ak4JetTracksAssociatorAtCaloFaceJPTPAT",
                       jet2TracksAtVX = "ak4JetTracksAssociatorAtVertexJPTPAT"
)

# ---------- Supported Modules

from CommonTools.RecoAlgos.trackFromPackedCandidateProducer_cfi import *
trackFromPackedCandidate = trackFromPackedCandidateProducer.clone(PFCandidates = 'packedPFCandidates')


from RecoJets.JetPlusTracks.jetPlusTrackAddonSeedProducer_cfi import *
JetPlusTrackAddonSeedRecoPAT = jetPlusTrackAddonSeedProducer.clone(
    srcCaloJets = "slimmedCaloJets",
    srcTrackJets = "ak4TrackJetsJPTPAT",
    srcPVs = 'offlineSlimmedPrimaryVertices',
    PFCandidates = 'packedPFCandidates',
    UsePAT = True
)


from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import *
trackWithVertexRefSelectorJPTPAT = trackWithVertexRefSelector.clone(
    vertexTag = 'offlineSlimmedPrimaryVertices',
    src = 'trackFromPackedCandidate'
)
from RecoJets.JetProducers.TracksForJets_cff import *
trackRefsForJetsJPTPAT = trackRefsForJets.clone(
    src = 'trackWithVertexRefSelectorJPTPAT'
)
from RecoJets.Configuration.RecoTrackJets_cff import *
ak4TrackJetsJPTPAT = ak4TrackJets.clone(
    srcPVs = 'offlineSlimmedPrimaryVertices',
    UseOnlyOnePV = True,
    src = 'trackRefsForJetsJPTPAT'
)

# ---------- Module definition

from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *

JetPlusTrackZSPCorJetAntiKt4PAT = cms.EDProducer(
    "JetPlusTrackProducer",
    cms.PSet(JPTZSPCorrectorAntiKt4),
    src = cms.InputTag("slimmedCaloJets"),
    srcTrackJets = cms.InputTag("ak4TrackJetsJPTPAT"), 
    srcAddCaloJets = cms.InputTag('JetPlusTrackAddonSeedRecoPAT'),
    extrapolations = cms.InputTag("trackExtrapolatorJPTPAT"),
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

# Task
PATJetPlusTrackCorrectionsAntiKt4Task = cms.Task(
    trackFromPackedCandidate,
    trackWithVertexRefSelectorJPTPAT,
    trackRefsForJetsJPTPAT,
    ak4TrackJetsJPTPAT,
    JetPlusTrackAddonSeedRecoPAT,
    trackExtrapolatorJPTPAT,
    ak4JetTracksAssociatorAtVertexJPTPAT,
    ak4JetTracksAssociatorAtCaloFaceJPTPAT,
    ak4JetExtenderJPTPAT,
    JetPlusTrackZSPCorJetAntiKt4PAT
    )

PATJetPlusTrackCorrectionsAntiKt4 = cms.Sequence(PATJetPlusTrackCorrectionsAntiKt4Task)

