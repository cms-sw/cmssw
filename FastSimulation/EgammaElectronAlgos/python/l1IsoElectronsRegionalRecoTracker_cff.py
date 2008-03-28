import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
l1IsoElectronsRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    #    untracked VInputTag RemoveTrackProducers = { 
    #	ctfL1IsoWithMaterialTracks
    #    }
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks"))
)

# The sequence
l1IsoElectronsRegionalRecoTracker = cms.Sequence(cms.SequencePlaceholder("globalPixelGSTracking")+l1IsoElectronsRegionalCTFFinalFitWithMaterial)

