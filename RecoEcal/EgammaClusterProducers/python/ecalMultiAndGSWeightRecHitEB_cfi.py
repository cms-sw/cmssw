import FWCore.ParameterSet.Config as cms

ecalMultiAndGSWeightRecHitEB = cms.EDProducer("CombinedRecHitCollectionProducer",
                                              primaryRecHits=cms.InputTag("reducedEcalRecHitsEB"),
                                              secondaryRecHits=cms.InputTag("ecalWeightRecHitSelectedDigis","EcalRecHitsEB"),
                                              outputCollectionName=cms.string(""),
                                              flagsToReplaceHit=cms.vstring("kHasSwitchToGain6","kHasSwitchToGain1")
                                              )
