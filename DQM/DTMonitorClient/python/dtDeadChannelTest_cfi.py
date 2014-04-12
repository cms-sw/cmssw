import FWCore.ParameterSet.Config as cms

deadChannelTest = cms.EDAnalyzer("DTDeadChannelTest",
    debug = cms.untracked.bool(False),
    runningStandalone = cms.untracked.bool(True),
    diagnosticPrescale = cms.untracked.int32(1),
    folderRoot = cms.untracked.string(''),
    #Names of the quality tests: they must match those specified in "qtList"
    EfficiencyTestName = cms.untracked.string('OccupancyDiffInRange')
)


