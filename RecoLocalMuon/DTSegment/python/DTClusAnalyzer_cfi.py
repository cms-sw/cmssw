import FWCore.ParameterSet.Config as cms

DTClusAnalyzer = cms.EDAnalyzer("DTClusAnalyzer",
    debug = cms.untracked.bool(True),
    rootFileName = cms.untracked.string('DTClusAnalyzer.root'),
    recClusLabel = cms.string('dt1DClusters'),
    recHits1DLabel = cms.string('dt1DRecHits'),
    recHits2DLabel = cms.string('dt2DSegments')
)



