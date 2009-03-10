import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
hltL1IsoEgammaRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForPhotons"),
                                   cms.InputTag("globalPixelTrackCandidatesForPhotons"))
)

# The sequence
HLTL1IsoEgammaRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                         hltL1IsoEgammaRegionalCTFFinalFitWithMaterial)

