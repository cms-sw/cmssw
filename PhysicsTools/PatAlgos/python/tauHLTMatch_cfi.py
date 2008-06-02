import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for muons
# (cuts are NOT tuned, using old values from TQAF MC match for e/mu, january 2008)
#
tauTrigMatchHLT1Tau = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Taus"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1Tau")
)

tauTrigMatchHLT2TauPixel = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Taus"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2TauPixel")
)

tauHLTMatch = cms.Sequence(tauTrigMatchHLT1Tau*tauTrigMatchHLT2TauPixel)

