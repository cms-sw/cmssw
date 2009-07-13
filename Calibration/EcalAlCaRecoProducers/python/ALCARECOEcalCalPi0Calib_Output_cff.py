import FWCore.ParameterSet.Config as cms

OutALCARECOEcalCalPi0Calib_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalPi0Calib')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ecalPi0Corrected_pi0EcalRecHitsEB_*',
        'keep *_ecalPi0Corrected_pi0EcalRecHitsEE_*',
        'keep *_hltAlCaPi0RegRecHits_pi0EcalRecHitsES_*')
)

import copy
OutALCARECOEcalCalPhiSym=copy.deepcopy(OutALCARECOEcalCalPhiSym_noDrop)
OutALCARECOEcalCalPhiSym.outputCommands.insert(0,"drop *")
