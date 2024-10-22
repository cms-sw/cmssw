import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCTriggerPrimitives.CSCCommonTrigger_cfi import *
csctfDigis = cms.EDProducer("CSCTFCandidateProducer",
    CSCTrackProducer = cms.untracked.InputTag("csctfTrackDigis"),
    MuonSorter = cms.PSet(
        CSCCommonTrigger
    )
)


