import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the L1 match for muons
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)
#
muonL1MatchL1SingleMu3 = cms.EDFilter("PATL1Matcher",
    src = cms.InputTag("allLayer0Muons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("L1SingleMu3")
)

muonL1Match = cms.Sequence(muonL1MatchL1SingleMu3)

