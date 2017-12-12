import FWCore.ParameterSet.Config as cms

pfCandidatesBadHadRecalibrated = cms.EDProducer("PFCandidateRecalibrator",
    pfcandidates = cms.InputTag("particleFlow")
)
