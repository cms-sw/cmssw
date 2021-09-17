import FWCore.ParameterSet.Config as cms

ecalPreshowerDataCertificationTask = cms.EDAnalyzer("ESDataCertificationTask",
    prefixME = cms.untracked.string('EcalPreshower'),
    mergeRuns = cms.untracked.bool(False)
)
