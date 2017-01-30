import FWCore.ParameterSet.Config as cms

#this does the simple fix (replace the crystals, assume fraction of 1)
gsSimpleFixedPhotons = cms.EDProducer("PhotonGSCrysSimpleFixer",
                                              oldPhos = cms.InputTag("gedPhotons"),
                                      ebMultiRecHits = cms.InputTag("reducedEcalRecHitsEB"),
                                      ebMultiAndWeightsRecHits = cms.InputTag("ecalMultiAndGSGlobalRecHitEB"),
                                      energyTypesToFix = cms.vstring("ecal_standard","ecal_photons","regression1","regression2"),
                                      energyTypeForP4 = cms.string("regression2")
)
 
