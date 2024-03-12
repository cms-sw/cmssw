import FWCore.ParameterSet.Config as cms

alcastreamEcalEtaCalibOutput = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_*_etaEcalRecHitsEB_*',
        'keep *_*_etaEcalRecHitsEE_*')
)

# foo bar baz
# S4MpPEhE7iP0P
# P7KvrJcWEhnXs
