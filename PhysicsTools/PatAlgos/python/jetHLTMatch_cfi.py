import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for muons
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)
#
jetHLTMatchHLT1Jet = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Jets"),
    maxDPtRel = cms.double(3.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1Jet")
)

jetHLTMatchHLT2Jet = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Jets"),
    maxDPtRel = cms.double(3.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2Jet")
)

jetHLTMatchHLT3Jet = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Jets"),
    maxDPtRel = cms.double(3.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT3Jet")
)

jetHLTMatchHLT4Jet = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Jets"),
    maxDPtRel = cms.double(3.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT4Jet")
)

jetHLTMatch = cms.Sequence(jetHLTMatchHLT1Jet*jetHLTMatchHLT2Jet*jetHLTMatchHLT3Jet*jetHLTMatchHLT4Jet)

