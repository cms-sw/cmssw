import FWCore.ParameterSet.Config as cms

ExceptionGenerator = cms.EDAnalyzer( "ExceptionGenerator",
             defaultAction       = cms.untracked.int32(-1),
             defaultQualifier    = cms.untracked.uint32(0)
)


