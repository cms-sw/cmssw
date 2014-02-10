import FWCore.ParameterSet.Config as cms

# ttbar dimuon
DiMuonHLTValidation = cms.EDAnalyzer('TopDiLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLTValidation/Top/DiMuon/'),
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
        sJets        = cms.untracked.string('ak5PFJets'),
        ptJets       = cms.untracked.double(30.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Mu17_Mu8','HLT_Mu17_TkMu8']),
)

# ttbar dielec
DiElectronHLTValidation = cms.EDAnalyzer('TopDiLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLTValidation/Top/DiElectron/'),
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
        sJets        = cms.untracked.string('ak5PFJets'),
        ptJets       = cms.untracked.double(30.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL']),
)

# ttbar elec-muon
ElecMuonHLTValidation = cms.EDAnalyzer('TopDiLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLTValidation/Top/ElecMuon/'),
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
        sJets        = cms.untracked.string('ak5PFJets'),
        ptJets       = cms.untracked.double(30.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL','HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL']),
)
