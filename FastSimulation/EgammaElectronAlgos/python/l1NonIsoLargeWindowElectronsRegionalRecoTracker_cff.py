import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    #    untracked VInputTag RemoveTrackProducers = { 
    #	ctfL1NonIsoLargeWindowWithMaterialTracks
    #    }
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"))
)

# The sequence
HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial)

