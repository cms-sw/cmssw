import FWCore.ParameterSet.Config as cms

efficiencyTest = cms.EDAnalyzer("DTEfficiencyTest",
    runningStandalone = cms.untracked.bool(True),
    UnassEfficiencyTestName = cms.untracked.string('UnassEfficiencyInRange'),
    #Names of the quality tests: they must match those specified in "qtList"
    EfficiencyTestName = cms.untracked.string('EfficiencyInRange'),
    folderRoot = cms.untracked.string(''),
    debug = cms.untracked.bool(False),
    diagnosticPrescale = cms.untracked.int32(1)
)


