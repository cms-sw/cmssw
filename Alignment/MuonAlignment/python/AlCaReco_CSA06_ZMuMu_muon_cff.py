import FWCore.ParameterSet.Config as cms

# AlCaReco for muon based alignment using ZMuMu events
dummyProcessor = cms.EDFilter("StopAfterNEvents",
    maxEvents = cms.int32(999999)
)

seqCSA06ZMuMu_muon = cms.Sequence(dummyProcessor)

