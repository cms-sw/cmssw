import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
l1IsoEgammaRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks"))
)

# The sequence
l1IsoEgammaRegionalRecoTracker = cms.Sequence(cms.SequencePlaceholder("globalPixelGSTracking")+l1IsoEgammaRegionalCTFFinalFitWithMaterial)

