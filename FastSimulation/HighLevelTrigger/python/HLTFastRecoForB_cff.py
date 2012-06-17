import FWCore.ParameterSet.Config as cms


from FastSimulation.Tracking.GlobalPixelTracking_cff import *

#############################################
# Reconstruct tracks with pixel seeds
#############################################

# Take all pixel tracks for b tagging track reco (pTMin>1GeV, nHits>=8)
hltFastTrackMerger = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0),
    minHits = cms.untracked.uint32(8)
)

hltBLifetimeRegionalCtfWithMaterialTracks = hltFastTrackMerger.clone()
hltBLifetimeRegionalCtfWithMaterialTracksSingleTop = hltFastTrackMerger.clone()
hltBLifetimeRegionalCtfWithMaterialTracksEleJetSingleTop = hltFastTrackMerger.clone()
hltBLifetimeRegionalCtfWithMaterialTracksIsoEleJetSingleTop = hltFastTrackMerger.clone()
hltBLifetimeRegionalCtfWithMaterialTracksRA2b = hltFastTrackMerger.clone()
hltBLifetimeRegionalCtfWithMaterialTracksRAzr = hltFastTrackMerger.clone()
hltBLifetimeRegionalCtfWithMaterialTracksHbb = hltFastTrackMerger.clone()
hltBLifetimeRegional3DCtfWithMaterialTracksJet30Hbb = hltFastTrackMerger.clone()
hltBLifetimeRegional3D1stTrkCtfWithMaterialTracksJet20Hbb = hltFastTrackMerger.clone()
hltBLifetimeRegional3DCtfWithMaterialTracksJet30Hbb = hltFastTrackMerger.clone()
hltBLifetimeBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20Hbb = hltFastTrackMerger.clone()
hltBLifetimeDiBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20Hbb = hltFastTrackMerger.clone()
hltBLifetimeRegionalCtfWithMaterialTracksbbPhi = hltFastTrackMerger.clone()
hltBLifetimeRegionalCtfWithMaterialTracksGammaB = hltFastTrackMerger.clone()

hltBLifetimeRegionalCkfTrackCandidates = cms.Sequence(globalPixelTracking)
hltBLifetimeRegionalCkfTrackCandidatesSingleTop = cms.Sequence(globalPixelTracking)
hltBLifetimeRegionalCkfTrackCandidatesEleJetSingleTop = cms.Sequence(globalPixelTracking)
hltBLifetimeRegionalCkfTrackCandidatesIsoEleJetSingleTop = cms.Sequence(globalPixelTracking)
hltBLifetimeRegionalCkfTrackCandidatesRA2b = cms.Sequence(globalPixelTracking)
hltBLifetimeRegionalCkfTrackCandidatesRAzr = cms.Sequence(globalPixelTracking)
hltBLifetimeRegionalCkfTrackCandidatesHbb = cms.Sequence(globalPixelTracking)
hltBLifetimeRegional3DCkfTrackCandidatesJet30Hbb = cms.Sequence(globalPixelTracking)
hltBLifetimeRegional3D1stTrkCkfTrackCandidatesJet20Hbb = cms.Sequence(globalPixelTracking)
hltBLifetimeRegional3DCkfTrackCandidatesJet30Hbb = cms.Sequence(globalPixelTracking)
hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20Hbb = cms.Sequence(globalPixelTracking)
hltBLifetimeDiBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20Hbb = cms.Sequence(globalPixelTracking)
hltBLifetimeRegionalCkfTrackCandidatesbbPhi = cms.Sequence(globalPixelTracking)
hltBLifetimeRegionalCkfTrackCandidatesGammaB = cms.Sequence(globalPixelTracking)


#############################################
# Reconstruct muons for MumuK
#############################################
import FWCore.ParameterSet.Config as cms

# Take all pixel-seeded tracks for b tagging track reco (pTMin>1GeV, nHits>=8) 
hltCtfWithMaterialTracksMumuk = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(3.0),
    minHits = cms.untracked.uint32(5)
)

# produce ChargedCandidates from tracks
hltMumukAllConeTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltCtfWithMaterialTracksMumuk"),
    particleType = cms.string('mu-')
)

hltCkfTrackCandidatesMumuk = cms.Sequence(cms.SequencePlaceholder("HLTL3muonrecoSequence"))


#############################################
# Reconstruct muons for JPsiToMumu
#############################################

# Take all pixel-seeded tracks for b tagging track reco (pTMin>1GeV, nHits>=8) 
hltCtfWithMaterialTracksMumu = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("hltL3Muons")),
    ptMin = cms.untracked.double(3.0),
    minHits = cms.untracked.uint32(5)
)

# produce ChargedCandidates from tracks
hltMuTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltCtfWithMaterialTracksMumu"),
    particleType = cms.string('mu-')
)

hltCkfTrackCandidatesMumu = cms.Sequence(cms.SequencePlaceholder("HLTL3muonrecoNocandSequence"))


