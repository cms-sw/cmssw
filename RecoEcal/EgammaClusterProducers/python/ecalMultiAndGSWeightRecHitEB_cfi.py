import FWCore.ParameterSet.Config as cms

ecalMultiAndGSWeightRecHitEB = cms.EDProducer("CombinedRecHitCollectionProducer",
                                              primaryRecHits=cms.InputTag("reducedEcalRecHitsEB"),
                                              secondaryRecHits=cms.InputTag("ecalWeightRecHitSelectedDigis","EcalRecHitsEB"),
                                              outputCollectionName=cms.string(""),
                                              outputReplacedHitsCollName =cms.string("gsMultiFit"),
                                              outputReplacingHitsCollName =cms.string("gsWeight"),
                                              
                                              flagsToReplaceHit=cms.vstring("kHasSwitchToGain6","kHasSwitchToGain1")
                                              )
