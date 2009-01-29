import FWCore.ParameterSet.Config as cms

## RECO-Truth standard matching criteria
tauTruthMatchingReqs = cms.PSet(
    maxDPtRel = cms.double(10000000.0),                 #don't apply a Pt resolution cut
    resolveByMatchQuality = cms.bool(True),
    resolveAmbiguities = cms.bool(True),
    maxDeltaR = cms.double(0.15)
)
qcdTruthMatchingReqs = cms.PSet(
    maxDPtRel = cms.double(10000000.0),
    resolveByMatchQuality = cms.bool(True),
    resolveAmbiguities = cms.bool(True),
    maxDeltaR = cms.double(0.3)
)

#########################
#  Tau Truth matchers   #
#########################

matchMCTausInsideOut = cms.EDProducer("PFTauDecayModeTruthMatcher",
    tauTruthMatchingReqs,
    src = cms.InputTag("makeMCTauDecayModes"),
    matched = cms.InputTag("pfRecoTauProducerInsideOut")
)

matchMCTausHighEfficiency = cms.EDProducer("PFTauDecayModeTruthMatcher",
    tauTruthMatchingReqs,
    src = cms.InputTag("makeMCTauDecayModes"),
    matched = cms.InputTag("pfRecoTauProducerHighEfficiency")
)

matchMCTaus = cms.Sequence(matchMCTausHighEfficiency)

#########################
#  QCD Truth matchers   #
#########################

matchMCQCDInsideOut = cms.EDProducer("PFTauDecayModeTruthMatcher",
    qcdTruthMatchingReqs,
    src = cms.InputTag("makeMCQCDTauDecayModes"),
    matched = cms.InputTag("pfRecoTauProducerInsideOut")
)

matchMCQCDHighEfficiency = cms.EDProducer("PFTauDecayModeTruthMatcher",
    qcdTruthMatchingReqs,
    src = cms.InputTag("makeMCQCDTauDecayModes"),
    matched = cms.InputTag("pfRecoTauProducerHighEfficiency")
)

matchMCQCD = cms.Sequence(matchMCQCDHighEfficiency)

