import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiPixelLA_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixelLorentzAngle')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiPixelLorentzAngle_*_*',
    )
)
OutALCARECOPromptCalibProdSiPixelLA=OutALCARECOPromptCalibProdSiPixelLA_noDrop.clone()
OutALCARECOPromptCalibProdSiPixelLA.outputCommands.insert(0, "drop *")
