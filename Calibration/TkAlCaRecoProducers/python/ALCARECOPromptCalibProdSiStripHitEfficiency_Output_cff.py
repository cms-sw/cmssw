import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiStripHitEfficiency_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiStripHitEfficiency')),
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConvertSiStripHitEff_*_*'))

OutALCARECOPromptCalibProdSiStripHitEfficiency = OutALCARECOPromptCalibProdSiStripHitEfficiency_noDrop.clone()
OutALCARECOPromptCalibProdSiStripHitEfficiency.outputCommands.insert(0, "drop *")
