import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    #    untracked VInputTag RemoveTrackProducers = { 
    #	ctfL1IsoLargeWindowWithMaterialTracks
    #    }
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates"))
)

# The sequence
HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial)

