import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the L1 match for muons
# (cuts are NOT tuned, using old values from TQAF MC match for jets, january 2008)
#
metL1MatchL1ETM20 = cms.EDFilter("PATL1Matcher",
    src = cms.InputTag("allLayer0METs"),
    maxDPtRel = cms.double(3.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("L1ETM20")
)

metL1Match = cms.Sequence(metL1MatchL1ETM20)

