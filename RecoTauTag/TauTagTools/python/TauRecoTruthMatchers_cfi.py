import FWCore.ParameterSet.Config as cms

tauTruthMatchingReqs = cms.PSet(
    maxDPtRel = cms.double(10000000.0),
    maxDeltaR = cms.double(0.15)
)
qcdTruthMatchingReqs = cms.PSet(
    maxDPtRel = cms.double(10000000.0),
    maxDeltaR = cms.double(0.3)
)
matchMCTausInsideOut = cms.EDProducer("PFTauDecayModeTruthMatcher",
    tauTruthMatchingReqs,
    resolveByMatchQuality = cms.bool(True),
    src = cms.InputTag("makeMCTauDecayModes"),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("pfRecoTauProducerInsideOut")
)

matchMCTausHighEfficiency = cms.EDProducer("PFTauDecayModeTruthMatcher",
    tauTruthMatchingReqs,
    resolveByMatchQuality = cms.bool(True),
    src = cms.InputTag("makeMCTauDecayModes"),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("pfRecoTauProducerHighEfficiency")
)

#matchMCTaus = cms.Sequence(matchMCTausHighEfficiency*matchMCTausInsideOut)

matchMCQCDInsideOut = cms.EDProducer("PFTauDecayModeTruthMatcher",
    qcdTruthMatchingReqs,
    resolveByMatchQuality = cms.bool(True),
    src = cms.InputTag("makeMCQCDTauDecayModes"),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("pfRecoTauProducerInsideOut")
)

matchMCQCDHighEfficiency = cms.EDProducer("PFTauDecayModeTruthMatcher",
    qcdTruthMatchingReqs,
    resolveByMatchQuality = cms.bool(True),
    src = cms.InputTag("makeMCQCDTauDecayModes"),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("pfRecoTauProducerHighEfficiency")
)

#matchMCQCD = cms.Sequence(matchMCQCDHighEfficiency*matchMCQCDInsideOut)

