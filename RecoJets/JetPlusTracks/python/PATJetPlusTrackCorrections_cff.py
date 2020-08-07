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
JetPlusTrackAddonSeedPAT = jetPlusTrackAddonSeedProducer.clone(
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
JetPlusTrackZSPCorJetAntiKt4PAT = JetPlusTrackZSPCorJetAntiKt4.clone(
    JetTracksAssociationAtVertex = 'ak4JetTracksAssociatorAtVertexJPTPAT',
    JetTracksAssociationAtCaloFace = 'ak4JetTracksAssociatorAtCaloFaceJPTPAT',
    Muons = 'slimmedMuons',
    Electrons = 'slimmedElectrons',
    JetSplitMerge = 2,
    UsePAT = True
)

### ---------- Sequences

# Task
PATJetPlusTrackCorrectionsAntiKt4Task = cms.Task(
    trackFromPackedCandidate,
    trackWithVertexRefSelectorJPTPAT,
    trackRefsForJetsJPTPAT,
    ak4TrackJetsJPTPAT,
    JetPlusTrackAddonSeedPAT,
    trackExtrapolatorJPTPAT,
    ak4JetTracksAssociatorAtVertexJPTPAT,
    ak4JetTracksAssociatorAtCaloFaceJPTPAT,
    ak4JetExtenderJPTPAT,
    JetPlusTrackZSPCorJetAntiKt4PAT
    )

PATJetPlusTrackCorrectionsAntiKt4 = cms.Sequence(PATJetPlusTrackCorrectionsAntiKt4Task)

