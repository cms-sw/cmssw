import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForElectrons"),
                                   cms.InputTag("globalPixelTrackCandidatesForElectrons"))
)

# The sequence
HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                                       hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial)

