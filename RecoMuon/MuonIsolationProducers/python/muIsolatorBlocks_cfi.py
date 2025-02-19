import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
IsolatorByDepositR03 = cms.PSet(
    IsolatorPSet = cms.PSet(
        ComponentName = cms.string('IsolatorByDeposit'),
        ConeSizeType = cms.string('FixedConeSize'),
        coneSize = cms.double(0.3)
    )
)


