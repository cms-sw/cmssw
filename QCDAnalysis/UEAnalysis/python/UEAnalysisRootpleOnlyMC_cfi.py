import FWCore.ParameterSet.Config as cms

UEAnalysisRootpleOnlyMC = cms.EDFilter("AnalysisRootpleProducerOnlyMC",
    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet"),
    MCEvent = cms.untracked.InputTag("generator"),
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    GammaGenPartCollectionName = cms.untracked.InputTag("gammaParticles"),
    GenJetCollectionName = cms.untracked.InputTag("IC5GenJet"),
    usegammaGen = cms.bool(False)
)

UEAnalysisRootpleOnlyMC500 = cms.EDFilter("AnalysisRootpleProducerOnlyMC",
    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet500"),
    MCEvent = cms.untracked.InputTag("generator"),
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    GammaGenPartCollectionName = cms.untracked.InputTag("gammaParticles"),
    GenJetCollectionName = cms.untracked.InputTag("IC5GenJet500"),
    usegammaGen = cms.bool(False)

)

UEAnalysisRootpleOnlyMC1500 = cms.EDFilter("AnalysisRootpleProducerOnlyMC",
    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet1500"),
    MCEvent = cms.untracked.InputTag("generator"),
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    GammaGenPartCollectionName = cms.untracked.InputTag("gammaParticles"),
    GenJetCollectionName = cms.untracked.InputTag("IC5GenJet1500"),
    usegammaGen = cms.bool(False)
)


UEAnalysisRootpleOnlyMC700 = cms.EDFilter("AnalysisRootpleProducerOnlyMC",
    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet700"),
    MCEvent = cms.untracked.InputTag("generator"),
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    GammaGenPartCollectionName = cms.untracked.InputTag("gammaParticles"),
    GenJetCollectionName = cms.untracked.InputTag("IC5GenJet700"),
    usegammaGen = cms.bool(False)
)

UEAnalysisRootpleOnlyMC1100 = cms.EDFilter("AnalysisRootpleProducerOnlyMC",
    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet1100"),
    MCEvent = cms.untracked.InputTag("generator"),
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    GammaGenPartCollectionName = cms.untracked.InputTag("gammaParticles"),
    GenJetCollectionName = cms.untracked.InputTag("IC5GenJet1100"),
    usegammaGen = cms.bool(False)
)



UEAnalysisOnlyMC = cms.Sequence(UEAnalysisRootpleOnlyMC*UEAnalysisRootpleOnlyMC500*UEAnalysisRootpleOnlyMC1500*UEAnalysisRootpleOnlyMC700*UEAnalysisRootpleOnlyMC1100)


