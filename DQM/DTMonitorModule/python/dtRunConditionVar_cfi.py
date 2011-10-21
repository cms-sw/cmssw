import FWCore.ParameterSet.Config as cms

dtRunConditionVar = cms.EDAnalyzer("DTRunConditionVar",
    debug = cms.untracked.bool(False),
    nMinHitsPhi = cms.untracked.int32(5),
    maxAnglePhiSegm = cms.untracked.double(30.),
    recoSegments = cms.InputTag("dt4DSegments"),                                 
)

