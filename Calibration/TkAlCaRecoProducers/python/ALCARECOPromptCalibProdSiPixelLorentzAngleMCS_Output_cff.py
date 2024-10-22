import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiPixelLAMCS_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixelLorentzAngleMCS')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiPixelLorentzAngleMCS_*_*',
    )
)
OutALCARECOPromptCalibProdSiPixelLAMCS=OutALCARECOPromptCalibProdSiPixelLAMCS_noDrop.clone()
OutALCARECOPromptCalibProdSiPixelLAMCS.outputCommands.insert(0, "drop *")
