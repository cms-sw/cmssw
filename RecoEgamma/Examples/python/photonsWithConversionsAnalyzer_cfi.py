import FWCore.ParameterSet.Config as cms

#
#  Author: N. Marinelli, U. of Notre Dame, US
#
photonsWithConversionsAnalyzer = cms.EDAnalyzer("PhotonsWithConversionsAnalyzer",
    phoProducer = cms.string('correctedPhotons'),
    HistOutFile = cms.untracked.string('analyzer.root'),
    moduleLabelMC = cms.untracked.string('source'),
    moduleLabelTk = cms.untracked.string('g4SimHits'),
    photonCollection = cms.string('correctedPhotonsWithConversions'),
    moduleLabelHit = cms.untracked.string('g4SimHits'),
    moduleLabelVtx = cms.untracked.string('g4SimHits')
)


