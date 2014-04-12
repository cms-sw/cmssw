import FWCore.ParameterSet.Config as cms

btagMuonInJet = cms.EDFilter("BTagSkimLeptonJet",
    CaloJet = cms.InputTag("iterativeCone5CaloJets"),
    MinimumCaloJetPt = cms.double(20.0),
    MinimumPtRel = cms.double(0.0),
    LeptonType = cms.string('muon'),
    Lepton = cms.InputTag("muons"),
    MinimumNLeptonJet = cms.int32(1),
    MaximumDeltaR = cms.double(0.4),
    MaximumLeptonEta = cms.double(2.5),
    MinimumLeptonPt = cms.double(6.0),
    MaximumCaloJetEta = cms.double(3.0)
)


