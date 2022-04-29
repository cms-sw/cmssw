import FWCore.ParameterSet.Config as cms

particleFlowBadHcalPseudoCluster = cms.EDProducer("PFBadHcalPseudoClusterProducer",
    debug = cms.untracked.bool(False),
    enable = cms.bool(False),
    mightGet = cms.optional.untracked.vstring
)
