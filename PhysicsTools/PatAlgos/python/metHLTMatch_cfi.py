import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for muons
# (cuts are NOT tuned, using old values from TQAF MC match for jets, january 2008)
#
metTrigMatchHLT1MET = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0METs"),
    maxDPtRel = cms.double(3.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1MET")
)

metHLTMatch = cms.Sequence(metTrigMatchHLT1MET)

