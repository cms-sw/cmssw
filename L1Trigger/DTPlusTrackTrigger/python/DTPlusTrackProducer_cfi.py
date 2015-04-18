import FWCore.ParameterSet.Config as cms

### Needed to access DTConfigManagerRcd and by DTTrig
from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff import *

DTPlusTrackProducer = cms.EDProducer(
    "DTPlusTrackProducer",

    TTStubs = cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
    TTTracks = cms.InputTag("TTTracksFromPixelDigis", "Level1TTTracks"),

    debug = cms.untracked.bool(False), # needed by DTTrig
    digiTag = cms.InputTag('simMuonDTDigis'), # needed by DTTrig

    useTSTheta = cms.untracked.bool(True),
    useRoughTheta = cms.untracked.bool(True),

    numSigmasForStubMatch = cms.untracked.double(4.),
    numSigmasForTkMatch = cms.untracked.double(3.),
    numSigmasForPtMatch = cms.untracked.double(3.),

    minL1TrackPt = cms.untracked.double(2.),

    minRInvB = cms.untracked.double( 0.00000045 ),
    maxRInvB = cms.untracked.double( 1.0 ),
    station2Correction = cms.untracked.double( 1.0 ),
    thirdMethodAccurate = cms.untracked.bool(False),
)

