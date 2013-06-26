import FWCore.ParameterSet.Config as cms

ecalPreshowerDcsInfoTask = cms.EDAnalyzer("ESDcsInfoTask",
    prefixME = cms.untracked.string('EcalPreshower'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    DcsStatusLabel = cms.InputTag("scalersRawToDigi")
)
