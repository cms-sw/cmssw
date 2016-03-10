import FWCore.ParameterSet.Config as cms




OutALCARECOPromptCalibProdSiStripGainsAfterAbortGap_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
                    'pathALCARECOPromptCalibProdSiStripGainsAfterAbortGap',
                                  )
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiStripGainsAfterAbortGap_*_*',
    )
)

import copy

OutALCARECOPromptCalibProdSiStripGainsAfterAbortGap=copy.deepcopy(OutALCARECOPromptCalibProdSiStripGainsAfterAbortGap_noDrop)
OutALCARECOPromptCalibProdSiStripGainsAfterAbortGap.outputCommands.insert(0, "drop *")
