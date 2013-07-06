import FWCore.ParameterSet.Config as cms

ueAnalysisRootple = cms.EDProducer("AnalysisRootpleProducer",
    #label of selected tracks
    TracksCollectionName = cms.untracked.InputTag("goodTracks"),
    #label of Jet made with Tracks
    TracksJetCollectionName = cms.untracked.InputTag("iterativeCone5BasicJetsSeed10"),
    #label of Jet made with only charged MC particles
    ChgGenJetCollectionName = cms.untracked.InputTag("iterativeCone5ChgGenJetsSeed10"),
    #lable of MC event
    MCEvent = cms.untracked.InputTag("source"),
    #label of charged MC particles
    ChgGenPartCollectionName = cms.untracked.InputTag("chargeParticles"),
    OnlyRECO = cms.untracked.bool(True),
    #label of standard Calo Jet 
    RecoCaloJetCollectionName = cms.untracked.InputTag("iterativeCone5CaloJets"),
    #label of Jet made with MC particles
    GenJetCollectionName = cms.untracked.InputTag("iterativeCone5GenJetsSeed10"),
    #label of trigger results
    triggerResults = cms.InputTag("TriggerResults")
)


