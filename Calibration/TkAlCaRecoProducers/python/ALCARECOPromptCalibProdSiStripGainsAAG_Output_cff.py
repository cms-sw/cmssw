import FWCore.ParameterSet.Config as cms




OutALCARECOPromptCalibProdSiStripGainsAAG_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
                    'pathALCARECOPromptCalibProdSiStripGainsAAG',
                                  )
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiStripGainsAAG_*_*',
    )
)

import copy

OutALCARECOPromptCalibProdSiStripGainsAAG=copy.deepcopy(OutALCARECOPromptCalibProdSiStripGainsAAG_noDrop)
OutALCARECOPromptCalibProdSiStripGainsAAG.outputCommands.insert(0, "drop *")
