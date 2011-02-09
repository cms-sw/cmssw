import FWCore.ParameterSet.Config as cms

OutALCARECOEcalCalEtaCalib_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalEtaCalib')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ecalEtaCorrected_etaEcalRecHitsEB_*',
        'keep *_ecalEtaCorrected_etaEcalRecHitsEE_*',
        'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
        'keep *_hltAlCaEtaRecHitsFilter_etaEcalRecHitsES_*')
)

import copy
OutALCARECOEcalCalEtaCalib=copy.deepcopy(OutALCARECOEcalCalEtaCalib_noDrop)
OutALCARECOEcalCalEtaCalib.outputCommands.insert(0,"drop *")
