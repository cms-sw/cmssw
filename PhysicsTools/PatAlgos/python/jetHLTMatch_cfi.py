import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for muons
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)
#
jetTrigMatchHLT1Jet = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Jets"),
    maxDPtRel = cms.double(3.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1Jet")
)

jetTrigMatchHLT2Jet = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Jets"),
    maxDPtRel = cms.double(3.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2Jet")
)

jetTrigMatchHLT3Jet = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Jets"),
    maxDPtRel = cms.double(3.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT3Jet")
)

jetTrigMatchHLT4Jet = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Jets"),
    maxDPtRel = cms.double(3.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT4Jet")
)

jetHLTMatch = cms.Sequence(jetTrigMatchHLT1Jet*jetTrigMatchHLT2Jet*jetTrigMatchHLT3Jet*jetTrigMatchHLT4Jet)

