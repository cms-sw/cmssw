import FWCore.ParameterSet.Config as cms

OutALCARECOEcalCalPi0Calib_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalPi0Calib')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ecalPi0Corrected_pi0EcalRecHitsEB_*',
        'keep *_ecalPi0Corrected_pi0EcalRecHitsEE_*',
        'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
        'keep *_hltAlCaPi0RecHitsFilter_pi0EcalRecHitsES_*')
)

import copy
OutALCARECOEcalCalPi0Calib=copy.deepcopy(OutALCARECOEcalCalPi0Calib_noDrop)
OutALCARECOEcalCalPi0Calib.outputCommands.insert(0,"drop *")
