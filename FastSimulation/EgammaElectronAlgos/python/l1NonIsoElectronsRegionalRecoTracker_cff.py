import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
l1NonIsoElectronsRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    #    untracked VInputTag RemoveTrackProducers = { 
    #	ctfL1NonIsoWithMaterialTracks
    #    }
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks"))
)

# The sequence
l1NonIsoElectronsRegionalRecoTracker = cms.Sequence(cms.SequencePlaceholder("globalPixelGSTracking")+l1NonIsoElectronsRegionalCTFFinalFitWithMaterial)

