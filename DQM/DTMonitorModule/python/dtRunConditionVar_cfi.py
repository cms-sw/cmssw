import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtRunConditionVar = DQMEDAnalyzer('DTRunConditionVar',
    debug = cms.untracked.bool(False),
    nMinHitsPhi = cms.untracked.int32(5),
    maxAnglePhiSegm = cms.untracked.double(30.),
    recoSegments = cms.InputTag('dt4DSegments'),                                 
)

