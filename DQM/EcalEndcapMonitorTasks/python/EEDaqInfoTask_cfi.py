import FWCore.ParameterSet.Config as cms

ecalEndcapDaqInfoTask = cms.EDAnalyzer("EEDaqInfoTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    EEMinusFedRangeMin = cms.untracked.int32(601),
    EEMinusFedRangeMax = cms.untracked.int32(609),
    EEPlusFedRangeMin = cms.untracked.int32(646),
    EEPlusFedRangeMax = cms.untracked.int32(654)
)
