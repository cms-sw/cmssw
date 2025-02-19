import FWCore.ParameterSet.Config as cms

chamberEfficiencyTest = cms.EDAnalyzer("DTChamberEfficiencyTest",
    runningStandalone = cms.untracked.bool(True),
    #Names of the quality tests: they must match those specified in "qtList"
    XEfficiencyTestName = cms.untracked.string('ChEfficiencyInRangeX'),
    YEfficiencyTestName = cms.untracked.string('ChEfficiencyInRangeY'),
    folderRoot = cms.untracked.string(''),
    debug = cms.untracked.bool(False),
    diagnosticPrescale = cms.untracked.int32(1)
)


