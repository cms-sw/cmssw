import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
hltL1SeededElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForElectrons"),
                                   cms.InputTag("globalPixelTrackCandidatesForElectrons"))
)

hltL1SeededStartUpElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForElectrons"),
                                   cms.InputTag("globalPixelTrackCandidatesForElectrons"))
)

# The sequence
HLTL1SeededElectronsRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                               hltL1SeededElectronsRegionalCTFFinalFitWithMaterial)
HLTL1SeededStartUpElectronsRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                                      hltL1SeededStartUpElectronsRegionalCTFFinalFitWithMaterial)

