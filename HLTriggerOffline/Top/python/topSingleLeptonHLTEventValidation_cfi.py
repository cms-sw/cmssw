import FWCore.ParameterSet.Config as cms

# ttbar semi muonique
topSingleMuonHLTValidation = cms.EDAnalyzer('TopSingleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLTValidation/Top/SemiMuonic/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(30.),
        etaElectrons = cms.untracked.double(2.5),
        isoElectrons = cms.untracked.double(0.1),
        minElectrons = cms.untracked.uint32(0),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(26.),
        etaMuons     = cms.untracked.double(2.1),
        isoMuons     = cms.untracked.double(0.12),
        minMuons     = cms.untracked.uint32(1),
        # Jets
        sJets        = cms.untracked.string('ak5PFJets'),
        ptJets       = cms.untracked.double(20.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(4),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet45_35_25','HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet30']),
)

# ttbar semi electronique
topSingleElectronHLTValidation = cms.EDAnalyzer('TopSingleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLTValidation/Top/SemiElectronic/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(30.),
        etaElectrons = cms.untracked.double(2.5),
        isoElectrons = cms.untracked.double(0.1),
        minElectrons = cms.untracked.uint32(1),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(26.),
        etaMuons     = cms.untracked.double(2.1),
        isoMuons     = cms.untracked.double(0.12),
        minMuons     = cms.untracked.uint32(0),
        # Jets
        sJets        = cms.untracked.string('ak5PFJets'),
        ptJets       = cms.untracked.double(20.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(4),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet45_35_25','HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet30']),
)
