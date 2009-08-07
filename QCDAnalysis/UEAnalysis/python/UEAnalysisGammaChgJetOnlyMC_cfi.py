import FWCore.ParameterSet.Config as cms

UEAnalysisRootpleOnlyMC = cms.EDFilter("AnalysisRootpleProducerOnlyMC",
    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet"),
    MCEvent = cms.untracked.InputTag("generator"),
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    GammaGenPartCollectionName = cms.untracked.InputTag("gammaParticles"),
    GenJetCollectionName = cms.untracked.InputTag("IC5GenJet")
)

UEAnalysisRootpleOnlyMC500 = cms.EDFilter("AnalysisRootpleProducerOnlyMC",
    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet500"),
    MCEvent = cms.untracked.InputTag("generator"),
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    GammaGenPartCollectionName = cms.untracked.InputTag("gammaParticles"),
    GenJetCollectionName = cms.untracked.InputTag("IC5GenJet500")
)

UEAnalysisOnlyMC = cms.Sequence(UEAnalysisRootpleOnlyMC*UEAnalysisRootpleOnlyMC500)
