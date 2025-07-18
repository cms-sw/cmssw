import FWCore.ParameterSet.Config as cms

hltInputLST = cms.EDProducer('LSTInputProducer@alpaka',
    ptCut = cms.double(0.8),
    phase2OTRecHits = cms.InputTag('hltSiPhase2RecHits'),
    beamSpot = cms.InputTag('hltOnlineBeamSpot'),
    seedTracks = cms.VInputTag(
      'hltInitialStepSeedTracksLST',
      'hltHighPtTripletStepSeedTracksLST'
    ),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)

_hltInputLSTSingleIterPatatrack = hltInputLST.clone(
    seedTracks = ['hltInitialStepSeedTracksLST']
)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
singleIterPatatrack.toReplaceWith(hltInputLST, _hltInputLSTSingleIterPatatrack)
