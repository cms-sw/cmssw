import FWCore.ParameterSet.Config as cms

# output block for alcastream EcalPhiSym
# keep L1 info and laser-recalibrated hits : 
# non-recalibrated hits have HLT as parent process

OutALCARECOEcalCalPhiSym_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalPhiSym')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ecalPhiSymCorrected_phiSymEcalRecHitsEB_*', 
        'keep *_ecalPhiSymCorrected_phiSymEcalRecHitsEE_*',
        'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*')
)


import copy
OutALCARECOEcalCalPhiSym=copy.deepcopy(OutALCARECOEcalCalPhiSym_noDrop)
OutALCARECOEcalCalPhiSym.outputCommands.insert(0,"drop *")
