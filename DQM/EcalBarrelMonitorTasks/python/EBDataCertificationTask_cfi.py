import FWCore.ParameterSet.Config as cms

ecalBarrelDataCertificationTask = cms.EDAnalyzer("EBDataCertificationTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)
