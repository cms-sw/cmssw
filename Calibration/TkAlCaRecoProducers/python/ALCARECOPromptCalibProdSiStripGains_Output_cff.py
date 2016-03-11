import FWCore.ParameterSet.Config as cms




OutALCARECOPromptCalibProdSiStripGains_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
                           'pathALCARECOPromptCalibProdSiStripGains')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiStripGains_*_*')
)

import copy

OutALCARECOPromptCalibProdSiStripGains=copy.deepcopy(OutALCARECOPromptCalibProdSiStripGains_noDrop)
OutALCARECOPromptCalibProdSiStripGains.outputCommands.insert(0, "drop *")
