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
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v',' HLT_Mu17_Mu8_v']),
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
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v','HLT_Ele23_Ele12_CaloId_TrackId_iso_v','HLT_Ele17_Ele8_Gsf_v']),
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
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Mu23_TrkIsoVVL_Ele12_Gsf_CaloId_TrackId_Iso_MediumWP_v','HLT_Mu8_TrkIsoVVL_Ele23_Gsf_CaloId_TrackId_Iso_MediumWP_v']),
)
