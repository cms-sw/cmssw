import FWCore.ParameterSet.Config as cms

btagElecInJet = cms.EDFilter("BTagSkimLeptonJet",
    CaloJet = cms.InputTag("iterativeCone5CaloJets"),
    MinimumCaloJetPt = cms.untracked.double(20.0),
    MinimumPtRel = cms.untracked.double(0.0),
    LeptonType = cms.untracked.string('electron'),
    Lepton = cms.InputTag("pixelMatchGsfElectrons"),
    MinimumNLeptonJet = cms.untracked.int32(1),
    MaximumDeltaR = cms.untracked.double(0.4),
    MaximumLeptonEta = cms.untracked.double(2.5),
    MinimumLeptonPt = cms.untracked.double(6.0),
    MaximumCaloJetEta = cms.untracked.double(3.0)
)


