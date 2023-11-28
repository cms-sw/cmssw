import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiStripLA_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiStripLorentzAngle')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiStripLorentzAngle_*_*',
    )
)
OutALCARECOPromptCalibProdSiStripLA=OutALCARECOPromptCalibProdSiStripLA_noDrop.clone()
OutALCARECOPromptCalibProdSiStripLA.outputCommands.insert(0, "drop *")
