import FWCore.ParameterSet.Config as cms

hltEgammaHcalIsol = cms.EDFilter("EgammaHLTHcalIsolationProducersRegional",
    hfRecHitProducer = cms.InputTag("hfreco"),
    recoEcalCandidateProducer = cms.InputTag("hltRecoEcalCandidate"),
    egHcalIsoPtMin = cms.double(0.0),
    egHcalIsoConeSize = cms.double(0.15),
    hbRecHitProducer = cms.InputTag("hbhereco")
)


