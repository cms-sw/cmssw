import FWCore.ParameterSet.Config as cms


chargedHadronPFTrackIsolation = cms.EDProducer(
    "ChargedHadronPFTrackIsolationProducer",
    src = cms.InputTag("particleFlow"),
    minTrackPt = cms.double(1),
    minRawCaloEnergy = cms.double(0.5),
    )
