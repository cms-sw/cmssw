import FWCore.ParameterSet.Config as cms

RPStraightTrackFitter = cms.EDAnalyzer("RPStraightTrackFitter",
    verbosity = cms.untracked.uint32(0),
    
    excludedRPs = cms.vuint32(),

    maxResidualToSigma = cms.double(3),
    minimumHitsPerProjectionPerRP = cms.uint32(4),

    dumpFileName = cms.untracked.string('trackFitDump.root')
)
