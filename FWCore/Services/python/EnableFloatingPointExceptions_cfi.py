import FWCore.ParameterSet.Config as cms

EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions",
    default = cms.untracked.PSet(
        enableOverFlowEx = cms.untracked.bool(False),
        enableDivByZeroEx = cms.untracked.bool(False),
        enableInvalidEx = cms.untracked.bool(False),
        enableUnderFlowEx = cms.untracked.bool(False)
    ),
    setPrecisionDouble = cms.untracked.bool(True),
    reportSettings = cms.untracked.bool(False),
    moduleNames = cms.untracked.vstring('default')
)


