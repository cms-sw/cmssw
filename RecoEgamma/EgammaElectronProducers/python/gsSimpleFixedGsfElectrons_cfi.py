import FWCore.ParameterSet.Config as cms 

#this does the simple fix (replace the crystals, assume fraction of 1)
gsSimpleFixedGsfElectrons = cms.EDProducer("GsfEleGSCrysSimpleFixer",
                                           oldEles = cms.InputTag("gedGsfElectrons"),
                                           ebMultiRecHits = cms.InputTag("reducedEcalRecHitsEB"),
                                           ebMultiAndWeightsRecHits = cms.InputTag("ecalMultiAndGSGlobalRecHitEB"),
)
