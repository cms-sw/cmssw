import FWCore.ParameterSet.Config as cms

slimmedSecondaryVertices = cms.EDProducer("PATSecondaryVertexSlimmer",
    src = cms.InputTag("inclusiveCandidateSecondaryVertices"),
    packedPFCandidates = cms.InputTag("packedPFCandidates")
)
# foo bar baz
# HZ2kBnJ31aHcl
# 1DTWotx8BjESr
