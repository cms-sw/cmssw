import FWCore.ParameterSet.Config as cms

import copy

# AlCaReco for Bad Component Identification
OutALCARECOEcalTestPulsesRaw_noDrop = cms.PSet(
    SelectEvents=cms.untracked.PSet(
        SelectEvents=cms.vstring('pathALCARECOEcalTestPulsesRaw')
    ),
    outputCommands=cms.untracked.vstring(
        'keep  FEDRawDataCollection_hltEcalCalibrationRaw_*_HLT',
        'keep  edmTriggerResults_*_*_HLT')
)

OutALCARECOEcalTestPulsesRaw = copy.deepcopy(OutALCARECOEcalTestPulsesRaw_noDrop)
OutALCARECOEcalTestPulsesRaw.outputCommands.insert(0, "drop *")
