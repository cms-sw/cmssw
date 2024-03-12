import FWCore.ParameterSet.Config as cms

btagElecInJet = cms.EDFilter("BTagSkimLeptonJet",
    CaloJet = cms.InputTag("iterativeCone5CaloJets"),
    MinimumCaloJetPt = cms.double(20.0),
    MinimumPtRel = cms.double(0.0),
    LeptonType = cms.string('electron'),
    Lepton = cms.InputTag("pixelMatchGsfElectrons"),
    MinimumNLeptonJet = cms.int32(1),
    MaximumDeltaR = cms.double(0.4),
    MaximumLeptonEta = cms.double(2.5),
    MinimumLeptonPt = cms.double(6.0),
    MaximumCaloJetEta = cms.double(3.0)
)


# foo bar baz
# 0TKhMYruvY3N6
# yiSilCLg3MsOE
