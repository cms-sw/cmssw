import FWCore.ParameterSet.Config as cms

DTEffAnalyzer = cms.EDAnalyzer("DTEffAnalyzer",
    recHits2DLabel = cms.string('dt2DSegments'),
    minHitsSegment = cms.int32(5),
    minCloseDist = cms.double(20.0),
    recHits4DLabel = cms.string('dt4DSegments'),
    rootFileName = cms.untracked.string('DTEffAnalyzer.root'),
    debug = cms.untracked.bool(False),
    recHits1DLabel = cms.string('dt1DRecHits'),
    minChi2NormSegment = cms.double(20.0)
)



