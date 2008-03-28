import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for electrons
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)
#
electronHLTMatchHLT1Electron = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Electrons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1Electron")
)

electronHLTMatchHLT1ElectronRelaxed = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Electrons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1ElectronRelaxed")
)

electronHLTMatchHLT2Electron = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Electrons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2Electron")
)

electronHLTMatchHLT2ElectronRelaxed = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Electrons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2ElectronRelaxed")
)

electronHLTMatch = cms.Sequence(electronHLTMatchHLT1Electron*electronHLTMatchHLT1ElectronRelaxed*electronHLTMatchHLT2Electron*electronHLTMatchHLT2ElectronRelaxed)

