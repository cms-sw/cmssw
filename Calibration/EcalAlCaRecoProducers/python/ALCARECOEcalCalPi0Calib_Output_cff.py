import FWCore.ParameterSet.Config as cms

OutALCARECOEcalCalPi0Calib = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalPi0Calib')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ecalPi0Corrected_pi0EcalRecHitsEB_*',
        'keep *_ecalPi0Corrected_pi0EcalRecHitsEE_*',
        'keep *_hltAlCaPi0RegRecHits_pi0EcalRecHitsES_*')
)

