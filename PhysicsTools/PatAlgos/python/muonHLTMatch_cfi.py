import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for muons
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)
#
muonHLTMatchHLT1MuonIso = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Muons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1MuonIso")
)

muonHLTMatchHLT1MuonNonIso = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Muons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1MuonNonIso")
)

muonHLTMatchHLT2MuonNonIso = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Muons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2MuonNonIso")
)

muonHLTMatch = cms.Sequence(muonHLTMatchHLT1MuonIso*muonHLTMatchHLT1MuonNonIso*muonHLTMatchHLT2MuonNonIso)

