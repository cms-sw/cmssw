import FWCore.ParameterSet.Config as cms




OutALCARECOPromptCalibProdSiStripGains_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
                           'pathALCARECOPromptCalibProdSiStripGainsIB',
                           'pathALCARECOPromptCalibProdSiStripGainsIB0T',
                           'pathALCARECOPromptCalibProdSiStripGainsAB',
                           'pathALCARECOPromptCalibProdSiStripGainsAB0T',
                                  )
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiStripGains_*_*',
        'keep *_MEtoEDMConvertSiStripGainsAllBunch_*_*',
        'keep *_MEtoEDMConvertSiStripGainsAllBunch0T_*_*',
        'keep *_MEtoEDMConvertSiStripGainsIsoBunch_*_*',
        'keep *_MEtoEDMConvertSiStripGainsIsoBunch0T_*_*')
)

import copy

OutALCARECOPromptCalibProdSiStripGains=copy.deepcopy(OutALCARECOPromptCalibProdSiStripGains_noDrop)
OutALCARECOPromptCalibProdSiStripGains.outputCommands.insert(0, "drop *")
