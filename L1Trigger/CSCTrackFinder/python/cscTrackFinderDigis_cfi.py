import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
cscTrackFinderDigis = cms.EDProducer("CSCTFCandidateProducer",
    CSCTrackProducer = cms.untracked.InputTag("l1CscTfTrackEmulDigis"),
    MuonSorter = cms.PSet(
        CSCCommonTrigger
    )
)


