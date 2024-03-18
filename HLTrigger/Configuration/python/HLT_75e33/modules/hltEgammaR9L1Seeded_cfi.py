import FWCore.ParameterSet.Config as cms

hltEgammaR9L1Seeded  = cms.EDProducer( "EgammaHLTR9IDProducer",
                                       ecalRechitEB = cms.InputTag("hltRechitInRegionsECAL","EcalRecHitsEB"),
                                       ecalRechitEE = cms.InputTag("hltRechitInRegionsECAL","EcalRecHitsEE"),
                                       recoEcalCandidateProducer = cms.InputTag( "hltEgammaCandidatesL1Seeded" )
)
