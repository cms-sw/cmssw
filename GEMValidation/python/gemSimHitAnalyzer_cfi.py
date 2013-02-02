import FWCore.ParameterSet.Config as cms


gemSimHitAnalyzer = cms.EDAnalyzer("GEMSimHitAnalyzer",
                                   minTrackPt = cms.double(10.0),
)
