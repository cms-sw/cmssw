import FWCore.ParameterSet.Config as cms

tTrigCalibrationTest = cms.EDAnalyzer("DTtTrigCalibrationTest",
    runningStandalone = cms.untracked.bool(True),
    tTrigTestName = cms.untracked.string('tTrigOffSet'),
    diagnosticPrescale = cms.untracked.int32(1),
    histoTag = cms.untracked.string('TimeBox'),
    #Names of the quality test: it must match those specified in "qtList"
    folderRoot = cms.untracked.string('')
)


