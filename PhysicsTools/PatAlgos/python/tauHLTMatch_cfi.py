import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for muons
# (cuts are NOT tuned, using old values from TQAF MC match for e/mu, january 2008)
#
tauHLTMatchHLT1Tau = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Taus"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1Tau")
)

tauHLTMatchHLT2TauPixel = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Taus"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2TauPixel")
)

tauHLTMatch = cms.Sequence(tauHLTMatchHLT1Tau*tauHLTMatchHLT2TauPixel)

