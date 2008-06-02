import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for muons
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)
#
muonTrigMatchHLT1MuonIso = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Muons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1MuonIso")
)

muonTrigMatchHLT1MuonNonIso = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Muons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1MuonNonIso")
)

muonTrigMatchHLT2MuonNonIso = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Muons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2MuonNonIso")
)

muonHLTMatch = cms.Sequence(muonTrigMatchHLT1MuonIso*muonTrigMatchHLT1MuonNonIso*muonTrigMatchHLT2MuonNonIso)

