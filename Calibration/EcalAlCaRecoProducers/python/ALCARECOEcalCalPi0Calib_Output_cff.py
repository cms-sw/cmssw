import FWCore.ParameterSet.Config as cms

OutALCARECOEcalCalPi0Calib = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalPi0Calib')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_*_pi0EcalRecHitsEB_*')
)

