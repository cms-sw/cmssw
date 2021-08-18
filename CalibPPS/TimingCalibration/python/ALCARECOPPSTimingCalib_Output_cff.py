import FWCore.ParameterSet.Config as cms

OutALCARECOPPSTimingCalib_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPPSTimingCalib')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_MEtoEDMConvertPPSTimingCalib_*_*',
    )
)

OutALCARECOPPSTimingCalib = OutALCARECOPPSTimingCalib_noDrop.clone()
OutALCARECOPPSTimingCalib.outputCommands.insert(0, 'drop *')
