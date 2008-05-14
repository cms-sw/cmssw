import FWCore.ParameterSet.Config as cms

# AlCaReco for muon calibration using MinBias events
OutALCARECOMuCaliMinBias = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuCaliMinBias')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_muonCSCDigis_*_*', 
        'keep *_muonDTDigis_*_*', 
        'keep *_muonRPCDigis_*_*')
)

