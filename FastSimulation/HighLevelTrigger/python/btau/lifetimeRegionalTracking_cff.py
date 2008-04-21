import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.GlobalPixelTracking_cff import *
hltBLifetimeRegionalCtfWithMaterialTracks = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks")),
    ptMin = cms.untracked.double(1.0),
    minHits = cms.untracked.uint32(8)
)

hltBLifetimeRegionalTracks = cms.Sequence(globalPixelGSTracking*hltBLifetimeRegionalCtfWithMaterialTracks)
hltBLifetimeL3tracking = cms.Sequence(hltBLifetimeRegionalTracks)

