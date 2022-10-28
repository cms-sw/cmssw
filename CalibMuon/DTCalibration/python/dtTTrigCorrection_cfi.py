import FWCore.ParameterSet.Config as cms

dtTTrigCorrection = cms.EDAnalyzer("DTTTrigCorrectionFirst",
    debug = cms.untracked.bool(False),
    dbLabel = cms.untracked.string(''),
    ttrigMax = cms.untracked.double(700.0),
    ttrigMin = cms.untracked.double(200.0),
    rmsLimit = cms.untracked.double(8.)
)
