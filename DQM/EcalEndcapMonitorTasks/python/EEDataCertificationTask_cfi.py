import FWCore.ParameterSet.Config as cms

ecalEndcapDataCertificationTask = cms.EDAnalyzer("EEDataCertificationTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)
