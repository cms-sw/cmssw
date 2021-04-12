import FWCore.ParameterSet.Config as cms

simCsctfDigis = cms.EDProducer("CSCTFCandidateProducer",
    CSCTrackProducer = cms.untracked.InputTag("simCsctfTrackDigis"),
    MuonSorter = cms.PSet(
        MaxBX = cms.int32(11),
        MinBX = cms.int32(5)
    )
)
