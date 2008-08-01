import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
l1CscTfEmulDigis = cms.EDProducer("CSCTFCandidateProducer",
    CSCTrackProducer = cms.untracked.InputTag("l1CscTfTrackEmulDigis"),
    MuonSorter = cms.PSet(
        CSCCommonTrigger
    )
)


