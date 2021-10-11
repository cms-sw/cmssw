import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiPixelLorentzAngle_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixelLorentzAngle')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiPixelLorentzAngle_*_*',
    )
)
OutALCARECOPromptCalibProdSiPixelLorentzAngle=OutALCARECOPromptCalibProdSiPixelLorentzAngle_noDrop.clone()
OutALCARECOPromptCalibProdSiPixelLorentzAngle.outputCommands.insert(0, "drop *")
