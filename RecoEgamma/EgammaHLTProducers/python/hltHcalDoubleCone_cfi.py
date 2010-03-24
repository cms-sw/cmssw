import FWCore.ParameterSet.Config as cms

hltHcalDoubleCone = cms.EDProducer("EgammaHLTHcalIsolationDoubleConeProducers",
    egHcalIsoConeSize = cms.double(0.3),
    hbRecHitProducer = cms.InputTag("hbhereco"),
    egHcalExclusion = cms.double(0.15),
    hfRecHitProducer = cms.InputTag("hfreco"),
    egHcalIsoPtMin = cms.double(0.0),
    recoEcalCandidateProducer = cms.InputTag("l1IsoRecoEcalCandidate")
)


