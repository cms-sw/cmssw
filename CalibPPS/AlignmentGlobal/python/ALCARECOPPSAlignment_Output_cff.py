import FWCore.ParameterSet.Config as cms

OutALCARECOPPSAlignment_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPPSAlignment')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_MEtoEDMConvertPPSAlignment_*_*',
    )
)

OutALCARECOPPSAlignment = OutALCARECOPPSAlignment_noDrop.clone()
OutALCARECOPPSAlignment.outputCommands.insert(0, 'drop *')
