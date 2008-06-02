import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for electrons
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)
#
electronTrigMatchHLT1Electron = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Electrons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1Electron")
)

electronTrigMatchHLT1ElectronRelaxed = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Electrons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1ElectronRelaxed")
)

electronTrigMatchHLT2Electron = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Electrons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2Electron")
)

electronTrigMatchHLT2ElectronRelaxed = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Electrons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2ElectronRelaxed")
)

electronHLTMatch = cms.Sequence(electronTrigMatchHLT1Electron*electronTrigMatchHLT1ElectronRelaxed*electronTrigMatchHLT2Electron*electronTrigMatchHLT2ElectronRelaxed)

