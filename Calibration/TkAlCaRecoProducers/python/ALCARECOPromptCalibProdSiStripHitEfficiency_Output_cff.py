import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiStripHitEff_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiStripHitEfficiency')),
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConvertSiStripHitEff_*_*'))

OutALCARECOPromptCalibProdSiStripHitEff = OutALCARECOPromptCalibProdSiStripHitEff_noDrop.clone()
OutALCARECOPromptCalibProdSiStripHitEff.outputCommands.insert(0, "drop *")
