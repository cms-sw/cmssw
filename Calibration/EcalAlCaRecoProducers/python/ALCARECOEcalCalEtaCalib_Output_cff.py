import FWCore.ParameterSet.Config as cms

OutALCARECOEcalCalEtaCalib = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalEtaCalib')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ecalEtaCorrected_etaEcalRecHitsEB_*',
        'keep *_ecalEtaCorrected_etaEcalRecHitsEE_*',
        'keep *_hltAlCaEtaRegRecHits_etaEcalRecHitsES_*')
)

