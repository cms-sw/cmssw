import FWCore.ParameterSet.Config as cms


OutALCARECOPromptCalibProdLumiPCC_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdLumiPCC')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_rawPCCProd_*_*',
        'keep *_corrPCCProd_*_*')
        )


import copy

OutALCARECOPromptCalibProdLumiPCC=copy.deepcopy(OutALCARECOPromptCalibProdLumiPCC_noDrop)
OutALCARECOPromptCalibProdLumiPCC.outputCommands.insert(0, "drop *")
