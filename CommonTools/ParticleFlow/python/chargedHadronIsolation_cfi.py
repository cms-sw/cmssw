import FWCore.ParameterSet.Config as cms


chargedHadronIsolation = cms.EDProducer(
    "ChargedHadronIsolationProducer",
    src = cms.InputTag("particleFlow"),
    minPt = cms.double(1),
    )
