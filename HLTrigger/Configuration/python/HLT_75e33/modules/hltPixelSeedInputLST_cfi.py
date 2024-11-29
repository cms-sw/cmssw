import FWCore.ParameterSet.Config as cms

hltPixelSeedInputLST = cms.EDProducer('LSTPixelSeedInputProducer',
    beamSpot = cms.InputTag('hltOnlineBeamSpot'),
    seedTracks = cms.VInputTag(
        'hltInitialStepSeedTracksLST',
        'hltHighPtTripletStepSeedTracksLST'
    )
)

_hltPixelSeedInputLSTSingleIterPatatrack = hltPixelSeedInputLST.clone(
    seedTracks = ['hltInitialStepSeedTracksLST']
)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
singleIterPatatrack.toReplaceWith(hltPixelSeedInputLST, _hltPixelSeedInputLSTSingleIterPatatrack)

