import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the L1 match for electrons/photons
# (cuts are NOT tuned, using old values from TQAF MC match for electrons, january 2008)
#
electronL1MatchL1SingleEG5 = cms.EDFilter("PATL1Matcher",
    src = cms.InputTag("allLayer0Electrons"),
    maxDPtRel = cms.double(0.5),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("L1SingleEG5")
)

photonL1MatchL1SingleEG5 = cms.EDFilter("PATL1Matcher",
    src = cms.InputTag("allLayer0Photons"),
    maxDPtRel = cms.double(1.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("L1SingleEG5")
)

electronL1Match = cms.Sequence(electronL1MatchL1SingleEG5)
photonL1Match = cms.Sequence(photonL1MatchL1SingleEG5)
emL1Match = cms.Sequence(electronL1Match*photonL1Match)

