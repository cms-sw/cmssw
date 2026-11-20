import FWCore.ParameterSet.Config as cms

l1tSC4PFL1PuppiCorrectedEmulatorMHT = cms.EDProducer("L1MhtPfProducer",
    jets = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulator"),
    maxJetEta = cms.double(2.4),
    minJetPt = cms.double(30.0)
)
