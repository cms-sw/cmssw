import FWCore.ParameterSet.Config as cms

ecalEndcapDaqInfoTask = cms.EDAnalyzer("EEDaqInfoTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
)

# ecalEndcapDaqInfoTask = cms.Sequence()
