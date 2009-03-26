import FWCore.ParameterSet.Config as cms

OutALCARECOEcalCalEtaCalib = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalEtaCalib')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_*_etaEcalRecHitsEB_*',
        'keep *_*_etaEcalRecHitsEE_*')
)

