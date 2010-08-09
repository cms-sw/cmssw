import FWCore.ParameterSet.Config as cms

L3MuonTrajectorySeedCombiner = cms.EDProducer("L3MuonTrajectorySeedCombiner",
    labels = cms.VInputTag()
)
