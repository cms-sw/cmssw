import FWCore.ParameterSet.Config as cms

hltEgammaClusterShapeL1Seeded = cms.EDProducer("EgammaHLTClusterShapeProducer",
    ecalRechitEB = cms.InputTag("hltRechitInRegionsECAL","EcalRecHitsEB"),
    ecalRechitEE = cms.InputTag("hltRechitInRegionsECAL","EcalRecHitsEE"),
    isIeta = cms.bool(True),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesL1Seeded")
)
