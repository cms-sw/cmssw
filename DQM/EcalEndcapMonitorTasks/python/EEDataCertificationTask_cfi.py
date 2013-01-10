import FWCore.ParameterSet.Config as cms

ecalEndcapDataCertificationTask = cms.EDAnalyzer("EEDataCertificationTask",
    cloneME = cms.untracked.bool(False),
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)
