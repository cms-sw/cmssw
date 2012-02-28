import FWCore.ParameterSet.Config as cms

ecalBarrelDataCertificationTask = cms.EDAnalyzer("EBDataCertificationTask",
    cloneME = cms.untracked.bool(False),
    prefixME = cms.untracked.string('Ecal'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)
