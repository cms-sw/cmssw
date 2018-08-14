import FWCore.ParameterSet.Config as cms


l1tPFMetNoMu = cms.EDProducer(
    "L1TPFMetNoMuProducer",
    pfMETCollection=cms.InputTag("pfMetT1"),
    muonCollection=cms.InputTag("muons"),
)
