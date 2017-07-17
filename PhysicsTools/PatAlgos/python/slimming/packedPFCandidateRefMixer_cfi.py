import FWCore.ParameterSet.Config as cms

packedPFCandidateRefMixer = cms.EDProducer("PackedPFCandidateRefMixer",
    pf = cms.InputTag("particleFlow"),
    pf2pf = cms.InputTag("FILLME"),
    pf2packed = cms.VInputTag(cms.InputTag("packedPFCandidates")),
)
