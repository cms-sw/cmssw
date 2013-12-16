import FWCore.ParameterSet.Config as cms

ueAnalysisRootple = cms.EDFilter("AnalysisRootpleProducerOnlyMC",
    #label of Jet made with only charged MC particles
    ChgGenJetCollectionName = cms.untracked.InputTag("ak4ChgGenJetsSeed10"),
    #label of MC event
    MCEvent = cms.untracked.InputTag("source"),
    #label of charged MC particles
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    #label of Jet made with MC particles
    GenJetCollectionName = cms.untracked.InputTag("ak4GenJetsSeed10")
)


