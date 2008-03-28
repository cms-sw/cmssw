import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
l1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    #    untracked VInputTag RemoveTrackProducers = { 
    #	ctfL1IsoLargeWindowWithMaterialTracks
    #    }
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks"))
)

# The sequence
l1IsoLargeWindowElectronsRegionalRecoTracker = cms.Sequence(cms.SequencePlaceholder("globalPixelGSTracking")+l1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial)

