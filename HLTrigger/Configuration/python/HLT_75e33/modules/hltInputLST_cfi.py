import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

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

hltInputLSTSerialSync = makeSerialClone(hltInputLST)

_hltInputLSTNGTScouting = hltInputLST.clone(
    seedTracks = ['hltInitialStepSeedTracksLST']
)

from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
ngtScouting.toReplaceWith(hltInputLST, _hltInputLSTNGTScouting )
