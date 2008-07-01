import FWCore.ParameterSet.Config as cms

# output block for alcastream EcalPhiSym
# keep L1 info and laser-recalibrated hits : 
# non-recalibrated hits have HLT as parent process
OutALCARECOEcalCalPhiSym = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalPhiSym')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_*_phiSymEcalRecHitsEB_*', 
        'keep *_*_phiSymEcalRecHitsEE_*',
        'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
        'keep *_MEtoEDMConverter_*_*')
)

