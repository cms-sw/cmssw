import FWCore.ParameterSet.Config as cms

trackFromPackedCandidateProducer = cms.EDProducer("TrackFromPackedCandidateProducer",
                                PFCandidates = cms.InputTag('packedPFCandidates')
)
