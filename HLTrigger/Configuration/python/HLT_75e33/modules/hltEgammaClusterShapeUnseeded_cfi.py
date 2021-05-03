import FWCore.ParameterSet.Config as cms

hltEgammaClusterShapeUnseeded = cms.EDProducer("EgammaHLTClusterShapeProducer",
    ecalRechitEB = cms.InputTag("hltEcalRecHit","EcalRecHitsEB"),
    ecalRechitEE = cms.InputTag("hltEcalRecHit","EcalRecHitsEE"),
    isIeta = cms.bool(True),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesUnseeded")
)
