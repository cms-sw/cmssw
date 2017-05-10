import FWCore.ParameterSet.Config as cms


chargedHadronIsolation = cms.EDProducer(
    "ChargedHadronIsolationProducer",
    src = cms.InputTag("particleFlow"),
    minTrackPt = cms.double(1),
    minRawCaloEnergy = cms.double(0.5),
    )
