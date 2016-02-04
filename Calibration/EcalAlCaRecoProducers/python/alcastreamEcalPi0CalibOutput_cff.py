import FWCore.ParameterSet.Config as cms

alcastreamEcalPi0CalibOutput = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_*_pi0EcalRecHitsEB_*')
)

