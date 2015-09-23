import FWCore.ParameterSet.Config as cms

# ttbar semi muonique
b2gSingleMuonHLTValidation = cms.EDAnalyzer('B2GSingleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/B2GHLTValidation/B2G/SemiMuonic/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(45.),
        etaElectrons = cms.untracked.double(2.5),
        minElectrons = cms.untracked.uint32(0),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(40.),
        etaMuons     = cms.untracked.double(2.1),
        minMuons     = cms.untracked.uint32(1),
        # Jets
        sJets        = cms.untracked.string('ak4PFJetsCHS'),
        ptJets0      = cms.untracked.double(200.),
        ptJets1      = cms.untracked.double(50.),
        etaJets      = cms.untracked.double(2.4),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Mu40_eta2p1_PFJet200_PFJet50',
		'HLT_Mu30_eta2p1_PFJet150_PFJet50']),
)

# ttbar semi electronique
b2gSingleElectronHLTValidation = cms.EDAnalyzer('B2GSingleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/B2GHLTValidation/B2G/SemiElectronic/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(45.),
        etaElectrons = cms.untracked.double(2.5),
        minElectrons = cms.untracked.uint32(1),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(40.),
        etaMuons     = cms.untracked.double(2.1),
        minMuons     = cms.untracked.uint32(0),
        # Jets
        sJets        = cms.untracked.string('ak4PFJetsCHS'),
        ptJets0      = cms.untracked.double(200.),
        ptJets1      = cms.untracked.double(50.),
        etaJets      = cms.untracked.double(2.4),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Ele45_CaloIdVT_GsfTrkIdT_PFJet200_PFJet50',
		'HLT_Ele35_CaloIdVT_GsfTrkIdT_PFJet150_PFJet50']),
)
