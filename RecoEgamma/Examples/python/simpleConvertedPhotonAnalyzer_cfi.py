import FWCore.ParameterSet.Config as cms

#
#  Author: N. Marinelli, U. of Notre Dame, US
#
convertedPhotonAnalyzer = cms.EDAnalyzer("SimpleConvertedPhotonAnalyzer",
    convertedPhotonCollection = cms.string('conversions'),
    phoProducer = cms.string('photonWithConversions'),
    HistOutFile = cms.untracked.string('analyzer.root'),
    moduleLabelMC = cms.untracked.string('source'),
    moduleLabelTk = cms.untracked.string('g4SimHits'),
    moduleLabelHit = cms.untracked.string('g4SimHits'),
    moduleLabelVtx = cms.untracked.string('g4SimHits')
)


