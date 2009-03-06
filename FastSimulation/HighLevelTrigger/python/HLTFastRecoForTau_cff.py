import FWCore.ParameterSet.Config as cms

#Primary Vertex
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
# (Not-so) Regional Tracking
from FastSimulation.Tracking.GlobalPixelTracking_cff import *
# L2.5 pixel-seeded tracks for SingleTau(MET) collections (pT>5GeV/c)
hltCtfWithMaterialTracksL25ElectronTau = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(5.0)
)

# L3 pixel-seeded tracks for Single Tau Collection (pT>1GeV/c)
hltCtfWithMaterialTracksL3SingleTau = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0)
)

hltCtfWithMaterialTracksL3SingleTauRelaxed = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0)
)

hltCtfWithMaterialTracksL3SingleTauMET = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0)
)

hltCtfWithMaterialTracksL3SingleTauMETRelaxed = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0)
)

hltCtfWithMaterialTracksL3ElectronTau = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0)
)

#--- Fastsim sequences replacing modules in tau paths ---#
hltCkfTrackCandidatesL3SingleTau = cms.Sequence(globalPixelTracking)
hltCkfTrackCandidatesL3SingleTauRelaxed = cms.Sequence(globalPixelTracking)
hltCkfTrackCandidatesL3SingleTauMET = cms.Sequence(globalPixelTracking)
hltCkfTrackCandidatesL3SingleTauMETRelaxed = cms.Sequence(globalPixelTracking)


