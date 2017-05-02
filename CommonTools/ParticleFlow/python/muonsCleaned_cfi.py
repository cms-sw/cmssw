import FWCore.ParameterSet.Config as cms

muonsCleaned = cms.EDProducer("PFMuonUntagger",
    muons = cms.InputTag("muons"),
    badmuons = cms.VInputTag(),
)
