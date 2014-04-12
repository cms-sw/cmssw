import FWCore.ParameterSet.Config as cms

dtRunConditionVarClient = cms.EDAnalyzer("DTRunConditionVarClient",

   minRangeVDrift  = cms.untracked.double(-1.),
   maxRangeVDrift  = cms.untracked.double(1.), 
   minRangeT0      = cms.untracked.double(-5.),
   maxRangeT0      = cms.untracked.double(5.),

   maxGoodVDriftDev = cms.untracked.double(0.006),
   minBadVDriftDev  = cms.untracked.double(0.018),

   maxGoodT0        = cms.untracked.double(2.),
   minBadT0         = cms.untracked.double(4.),

   maxGoodVDriftSigma = cms.untracked.double(0.0001),
   minBadVDriftSigma  = cms.untracked.double(0.001),

   maxGoodT0Sigma     = cms.untracked.double(2.),
   minBadT0Sigma      = cms.untracked.double(4.),

)
