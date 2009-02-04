import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForElectrons"),
                                   cms.InputTag("globalPixelTrackCandidatesForElectrons"))
)

hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForElectrons"),
                                   cms.InputTag("globalPixelTrackCandidatesForElectrons"))
)

# The sequence
HLTL1NonIsoElectronsRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                               hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial)
HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                                      hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial)

