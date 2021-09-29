import FWCore.ParameterSet.Config as cms

hltPFClusterMET = cms.EDProducer("PFClusterMETProducer",
    alias = cms.string('pfClusterMet'),
    globalThreshold = cms.double(0.0),
    src = cms.InputTag("pfClusterRefsForJets")
)
