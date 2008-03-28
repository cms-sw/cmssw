import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
l1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    #    untracked VInputTag RemoveTrackProducers = { 
    #	ctfL1NonIsoLargeWindowWithMaterialTracks
    #    }
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks"))
)

# The sequence
l1NonIsoLargeWindowElectronsRegionalRecoTracker = cms.Sequence(cms.SequencePlaceholder("globalPixelGSTracking")+l1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial)

