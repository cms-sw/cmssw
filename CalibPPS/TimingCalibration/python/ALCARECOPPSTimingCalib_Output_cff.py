import FWCore.ParameterSet.Config as cms
import copy

OutALCARECOPPSTimingCalib_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPPSTimingCalib')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_MEtoEDMConvertPPSTimingCalib_*_*',
    )
)

OutALCARECOPPSTimingCalib = copy.deepcopy(OutALCARECOPPSTimingCalib_noDrop)
OutALCARECOPPSTimingCalib.outputCommands.insert(0, 'drop *')
