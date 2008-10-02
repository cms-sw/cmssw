import FWCore.ParameterSet.Config as cms

UEAnalysisRootple = cms.EDProducer("AnalysisRootpleProducer",
    TracksCollectionName = cms.untracked.InputTag("goodTracks"),
    RecoCaloJetCollectionName = cms.untracked.InputTag("iterativeCone5CaloJets"),
    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet"),
    MCEvent = cms.untracked.InputTag("source"),
    TracksJetCollectionName = cms.untracked.InputTag("IC5TracksJet"),
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD"),
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    OnlyRECO = cms.untracked.bool(True),
    GenJetCollectionName = cms.untracked.InputTag("IC5GenJet"),
    triggerResults = cms.InputTag("TriggerResults","","HLT")
)

UEAnalysisRootple500 = cms.EDProducer("AnalysisRootpleProducer",
    TracksCollectionName = cms.untracked.InputTag("goodTracks"),
    RecoCaloJetCollectionName = cms.untracked.InputTag("iterativeCone5CaloJets"),
    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet500"),
    MCEvent = cms.untracked.InputTag("source"),
    TracksJetCollectionName = cms.untracked.InputTag("IC5TracksJet500"),
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD"),
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    OnlyRECO = cms.untracked.bool(True),
    GenJetCollectionName = cms.untracked.InputTag("IC5GenJet500"),
    triggerResults = cms.InputTag("TriggerResults","","HLT")
)

UEAnalysisRootple1500 = cms.EDProducer("AnalysisRootpleProducer",
    TracksCollectionName = cms.untracked.InputTag("goodTracks"),
    RecoCaloJetCollectionName = cms.untracked.InputTag("iterativeCone5CaloJets"),
    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet1500"),
    MCEvent = cms.untracked.InputTag("source"),
    TracksJetCollectionName = cms.untracked.InputTag("IC5TracksJet1500"),
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD"),
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    OnlyRECO = cms.untracked.bool(True),
    GenJetCollectionName = cms.untracked.InputTag("IC5GenJet1500"),
    triggerResults = cms.InputTag("TriggerResults","","HLT")
)

UEAnalysis = cms.Sequence(UEAnalysisRootple*UEAnalysisRootple500*UEAnalysisRootple1500)


