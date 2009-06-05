import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForElectrons"),
                                   cms.InputTag("globalPixelTrackCandidatesForElectrons"))
)

# The sequence
HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                                          hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial)

