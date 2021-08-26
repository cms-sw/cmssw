import FWCore.ParameterSet.Config as cms

OutALCARECOPPSDiamondSampicTimingCalib_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPPSDiamondSampicTimingCalib')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_MEtoEDMConvertPPSDiamondSampicTimingCalib_*_*',
    )
)

OutALCARECOPPSDiamondSampicTimingCalib = OutALCARECOPPSDiamondSampicTimingCalib_noDrop.clone()
OutALCARECOPPSDiamondSampicTimingCalib.outputCommands.insert(0, 'drop *')
