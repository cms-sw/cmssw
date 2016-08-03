import FWCore.ParameterSet.Config as cms

# ttbar dimuon
DiMuonHLTValidation = cms.EDAnalyzer('TopDiLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/TopHLTValidation/Top/DiMuon/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(20.),
        etaElectrons = cms.untracked.double(2.5),
        isoElectrons = cms.untracked.double(0.15),
        minElectrons = cms.untracked.uint32(0),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(20.),
        etaMuons     = cms.untracked.double(2.4),
        isoMuons     = cms.untracked.double(0.2),
        minMuons     = cms.untracked.uint32(2),
        # Jets
        sJets        = cms.untracked.string('ak4PFJetsCHS'),
        ptJets       = cms.untracked.double(30.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        iTrigger     = cms.untracked.InputTag("TriggerResults","","HLT"),
### Updating to  HLT paths to be monitored by TOP PAG in 2016                                                                                                                
		vsPaths      = cms.untracked.vstring(['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v']),
)

# ttbar dielec
DiElectronHLTValidation = cms.EDAnalyzer('TopDiLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/TopHLTValidation/Top/DiElectron/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(20.),
        etaElectrons = cms.untracked.double(2.5),
        isoElectrons = cms.untracked.double(0.15),
        minElectrons = cms.untracked.uint32(2),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(20.),
        etaMuons     = cms.untracked.double(2.4),
        isoMuons     = cms.untracked.double(0.2),
        minMuons     = cms.untracked.uint32(0),
        # Jets
        sJets        = cms.untracked.string('ak4PFJetsCHS'),
        ptJets       = cms.untracked.double(30.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        iTrigger     = cms.untracked.InputTag("TriggerResults","","HLT"),
### Updating to HLT paths to be monitored by TOP PAG in 2016                                                                                                                 
		vsPaths      = cms.untracked.vstring(['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v', 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v']),
)

# ttbar elec-muon
ElecMuonHLTValidation = cms.EDAnalyzer('TopDiLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/TopHLTValidation/Top/ElecMuon/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(20.),
        etaElectrons = cms.untracked.double(2.5),
        isoElectrons = cms.untracked.double(0.15),
        minElectrons = cms.untracked.uint32(1),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(20.),
        etaMuons     = cms.untracked.double(2.4),
        isoMuons     = cms.untracked.double(0.2),
        minMuons     = cms.untracked.uint32(1),
        # Jets
        sJets        = cms.untracked.string('ak4PFJetsCHS'),
        ptJets       = cms.untracked.double(30.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        iTrigger     = cms.untracked.InputTag("TriggerResults","","HLT"),
### Updating to HLT paths to be monitored by TOP PAG in 2016                                                                                                                 
		vsPaths      = cms.untracked.vstring(['HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v', 'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v', 'HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v', 'HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v']),
)
