import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for photons
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)
#
photonHLTMatchHLT1Photon = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Photons"),
    maxDPtRel = cms.double(1.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1Photon")
)

photonHLTMatchHLT1PhotonRelaxed = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Photons"),
    maxDPtRel = cms.double(1.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1PhotonRelaxed")
)

photonHLTMatchHLT2Photon = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Photons"),
    maxDPtRel = cms.double(1.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2Photon")
)

photonHLTMatchHLT2PhotonRelaxed = cms.EDFilter("PATHLTMatcher",
    src = cms.InputTag("allLayer0Photons"),
    maxDPtRel = cms.double(1.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2PhotonRelaxed")
)

photonHLTMatch = cms.Sequence(photonHLTMatchHLT1Photon*photonHLTMatchHLT1PhotonRelaxed*photonHLTMatchHLT2Photon*photonHLTMatchHLT2PhotonRelaxed)

