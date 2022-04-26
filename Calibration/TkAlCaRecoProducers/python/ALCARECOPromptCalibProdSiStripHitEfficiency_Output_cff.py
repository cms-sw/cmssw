import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiStripHitEfficiency_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiStripHitEfficiency')),
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConvertSiStripHitEff_*_*'))

import copy
OutALCARECOPromptCalibProdSiStripHitEfficiency=copy.deepcopy(OutALCARECOPromptCalibProdSiStripHitEfficiency_noDrop)
OutALCARECOPromptCalibProdSiStripHitEfficiency.outputCommands.insert(0, "drop *")
