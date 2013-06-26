import FWCore.ParameterSet.Config as cms

ecalPreshowerDataCertificationTask = cms.EDAnalyzer("ESDataCertificationTask",
    prefixME = cms.untracked.string('EcalPreshower'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)
