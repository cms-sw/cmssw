import FWCore.ParameterSet.Config as cms

HardwareMonitor = cms.EDAnalyzer("CnBAnalyzer",
    preSwapOn = cms.untracked.bool(True), ## hackswap the FED header

    rootFile = cms.untracked.string('FED_DQM_Data.root'),
    swapOn = cms.untracked.bool(True), ## non zero value does the DAQ header offset, etc.

    buildAllHistograms = cms.untracked.bool(False),
    rootFileDirectory = cms.untracked.string('/tmp')
)


