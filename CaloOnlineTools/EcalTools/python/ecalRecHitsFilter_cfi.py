import FWCore.ParameterSet.Config as cms

ecalRecHitsFilter = cms.EDFilter("EcalRecHitsFilter",
    
    EcalRecHitCollectionEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EcalRecHitCollectionEE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),

    # parameter for the name of the output root file with TH1F
    fileName = cms.untracked.string('ecalRecHitsFilterHists'),
    NumberXtalsThreshold = cms.untracked.int32(20),
    energycut = cms.untracked.double(0.2)

)
# foo bar baz
# M0Yr6uqxV5mkb
# xLJhSo2F5DD1T
