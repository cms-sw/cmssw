import FWCore.ParameterSet.Config as cms

IsoTrigHB = cms.EDAnalyzer("IsoTrig",
                           Det           = cms.string("HB"),
                           Verbosity     = cms.untracked.int32( 0 ),
                         
                           TrackQuality    = cms.untracked.string("highPurity"),
                           MinTrackPt      = cms.untracked.double(10.0),
                           MaxDxyPV        = cms.untracked.double(0.02),
                           MaxDzPV         = cms.untracked.double(0.02),
                           MaxChi2         = cms.untracked.double(5.0),
                           MaxDpOverP      = cms.untracked.double(0.1),
                           MinOuterHit     = cms.untracked.int32(4),
                           MinLayerCrossed =cms.untracked.int32(8),
                           MaxInMiss       = cms.untracked.int32(0),
                           MaxOutMiss      = cms.untracked.int32(0),
                           IsolationL1     = cms.untracked.double(1.0),
                           ConeRadius      = cms.untracked.double(34.98),
                           ConeRadiusMIP   = cms.untracked.double(14.0),
                           ConeRadiusNeut1 = cms.untracked.double(21.0),
                           ConeRadiusNeut2 = cms.untracked.double(29.0),
                           MIPCut          = cms.untracked.double(1.0),
                           ChargeIsolation = cms.untracked.double(2.0),
                           NeutralIsolation= cms.untracked.double(2.0),
                           minRun          =cms.untracked.int32(190456),
                           maxRun          =cms.untracked.int32(203002)
)
