import FWCore.ParameterSet.Config as cms

# The tracks (Ttake all pixel tracks but the potential electrons):
hltEcalActivityEgammaRegionalCTFFinalFitWithMaterial = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForPhotons"),
                                   cms.InputTag("globalPixelTrackCandidatesForPhotons"))
)

hltCtfActivityWithMaterialTracks = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracksForPhotons"),
                                   cms.InputTag("globalPixelTrackCandidatesForPhotons"))
) 


# The sequences
HLTEcalActivityEgammaRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                            hltEcalActivityEgammaRegionalCTFFinalFitWithMaterial)

HLTPixelMatchElectronActivityTrackingSequence  = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")+
                                                            hltCtfActivityWithMaterialTracks)  
