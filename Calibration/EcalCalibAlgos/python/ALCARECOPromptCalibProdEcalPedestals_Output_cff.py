import FWCore.ParameterSet.Config as cms
import copy

OutALCARECOPromptCalibProdEcalPedestals_noDrop = cms.PSet(
    SelectEvents=cms.untracked.PSet(
        SelectEvents=cms.vstring(
                           'pathALCARECOPromptCalibProdEcalPedestals')
    ),
    outputCommands=cms.untracked.vstring(
        'keep *_MEtoEDMConvertEcalPedestals_*_*')
)

OutALCARECOPromptCalibProdEcalPedestals = copy.deepcopy(OutALCARECOPromptCalibProdEcalPedestals_noDrop)
OutALCARECOPromptCalibProdEcalPedestals.outputCommands.insert(0, "drop *")
