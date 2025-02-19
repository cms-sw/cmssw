import FWCore.ParameterSet.Config as cms

hltHcalNoiseFilter = cms.EDFilter("HLTHcalNoiseFilter",
        JetSource = cms.InputTag("iterativeCone5CaloJets"),
        MetSource = cms.InputTag("met"),
        TowerSource = cms.InputTag("towerMaker"),
	UseJet = cms.bool(True),
	UseMET = cms.bool(False),
        MetCut = cms.double(0),
        JetMinE = cms.double(20),
        JetHCALminEnergyFraction = cms.double(0.98),
        saveTags = cms.bool( False )
)

