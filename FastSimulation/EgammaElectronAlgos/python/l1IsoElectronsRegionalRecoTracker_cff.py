import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
hltL1IsoElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForElectrons"),
                                   cms.InputTag("globalPixelTrackCandidatesForElectrons"))
)

hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForElectrons"),
                                   cms.InputTag("globalPixelTrackCandidatesForElectrons"))
)

# The sequence
HLTL1IsoElectronsRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                            hltL1IsoElectronsRegionalCTFFinalFitWithMaterial)

HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                                   hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial)

