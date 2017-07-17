import FWCore.ParameterSet.Config as cms

IsoTrackCalib = cms.EDAnalyzer("IsoTrackCalib",
                               Verbosity       = cms.untracked.int32( 0 ),
                               L1Seed          = cms.untracked.vstring("L1_SingleJet36","L1_SingleJet52","L1_SingleJet68","L1_SingleJet92","L1_SingleJet128"),
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
                               ConeRadius      = cms.untracked.double(34.98),
                               ConeRadiusMIP   = cms.untracked.double(14.0),
                               IsItAOD         = cms.untracked.bool(False),
)
