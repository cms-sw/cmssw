import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the HLT match for photons
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)
#
photonTrigMatchHLT1Photon = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Photons"),
    maxDPtRel = cms.double(1.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1Photon")
)

photonTrigMatchHLT1PhotonRelaxed = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Photons"),
    maxDPtRel = cms.double(1.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT1PhotonRelaxed")
)

photonTrigMatchHLT2Photon = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Photons"),
    maxDPtRel = cms.double(1.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2Photon")
)

photonTrigMatchHLT2PhotonRelaxed = cms.EDFilter("PATTrigMatcher",
    src = cms.InputTag("allLayer0Photons"),
    maxDPtRel = cms.double(1.0),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("HLT2PhotonRelaxed")
)

photonHLTMatch = cms.Sequence(photonTrigMatchHLT1Photon*photonTrigMatchHLT1PhotonRelaxed*photonTrigMatchHLT2Photon*photonTrigMatchHLT2PhotonRelaxed)

