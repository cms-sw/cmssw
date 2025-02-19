import FWCore.ParameterSet.Config as cms

isolatedTracksHcal = cms.EDAnalyzer("IsolatedTracksHcalScale",
                                    doMC          = cms.untracked.bool(False),
                                    Verbosity     = cms.untracked.int32( 0 ),
                                    
                                    TrackQuality  = cms.untracked.string("highPurity"),
                                    MinTrackPt    = cms.untracked.double(10.0),
                                    MaxDxyPV      = cms.untracked.double(0.02),
                                    MaxDzPV       = cms.untracked.double(0.02),
                                    MaxChi2       = cms.untracked.double(5.0),
                                    MaxDpOverP    = cms.untracked.double(0.1),
                                    MinOuterHit   = cms.untracked.int32(4),
                                    MinLayerCrossed=cms.untracked.int32(8),
                                    MaxInMiss     = cms.untracked.int32(0),
                                    MaxOutMiss    = cms.untracked.int32(0),
                                    ConeRadius    = cms.untracked.double(34.98),
                                    ConeRadiusMIP = cms.untracked.double(14.0),
                                    TimeMinCutECAL= cms.untracked.double(-500),
                                    TimeMaxCutECAL= cms.untracked.double(500)
)
