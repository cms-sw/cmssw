import FWCore.ParameterSet.Config as cms

#Primary Vertex
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
# (Not-so) Regional Tracking
from FastSimulation.Tracking.GlobalPixelTracking_cff import *

# L3 pixel-seeded tracks for Single Tau Collection (pT>1GeV/c)
hltL3TauCtfWithMaterialTracks = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0)

)
hltL3TauCtfWithMaterialHighPtTracks = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0)
)
hltL25TauCtfWithMaterialTracks = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0)
)
hltL3TauSingleTrack15CtfWithMaterialTracks = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(0.9)
)
hltPFJetCtfWithMaterialTracks = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(0.2)
)

#--- Fastsim sequences replacing modules in tau paths ---#
#hltL3TauCkfTrackCandidates = cms.Sequence(globalPixelTracking)
HLTL3TauTrackReconstructionSequence = cms.Sequence(globalPixelTracking + hltL3TauCtfWithMaterialTracks)
HLTL3TauHighPtTrackReconstructionSequence = cms.Sequence(globalPixelTracking + hltL3TauCtfWithMaterialHighPtTracks)
HLTL25TauTrackReconstructionSequence = cms.Sequence(globalPixelTracking + hltL25TauCtfWithMaterialTracks)
HLTL3TauSingleTrack15ReconstructionSequence = cms.Sequence(globalPixelTracking + hltL3TauSingleTrack15CtfWithMaterialTracks)
HLTTrackReconstructionForJets  = cms.Sequence(globalPixelTracking + hltPFJetCtfWithMaterialTracks)
