import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
hltEcalActivityEgammaRegionalCTFFinalFitWithMaterial = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForPhotons"),
                                   cms.InputTag("globalPixelTrackCandidatesForPhotons"))
)

# The sequence
HLTEcalActivityEgammaRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                            hltEcalActivityEgammaRegionalCTFFinalFitWithMaterial)

