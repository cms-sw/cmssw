import FWCore.ParameterSet.Config as cms

ecalPreshowerDcsInfoTask = cms.EDAnalyzer("ESDcsInfoTask",
    prefixME = cms.untracked.string('EcalPreshower'),
    mergeRuns = cms.untracked.bool(False),
    DcsStatusLabel = cms.InputTag("scalersRawToDigi")
)
# foo bar baz
# f6AJcu7pFqef8
