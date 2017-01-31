import FWCore.ParameterSet.Config as cms

ecalMultiAndGSGlobalRecHitEB = cms.EDProducer("CombinedRecHitCollectionProducer",
                                              primaryRecHits=cms.InputTag("reducedEcalRecHitsEB",processName=cms.InputTag.skipCurrentProcess()),
                                              secondaryRecHits=cms.InputTag("ecalGlobalRecHitSelectedDigis","EcalRecHitsEB"),
                                              outputCollectionName=cms.string(""),
                                              outputReplacedHitsCollName =cms.string("gsMultiFit"),
                                              outputReplacingHitsCollName =cms.string("gsGlobal"),                                              
                                              flagsToReplaceHit=cms.vstring("kHasSwitchToGain6","kHasSwitchToGain1")
                                              )
