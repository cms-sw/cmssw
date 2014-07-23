import FWCore.ParameterSet.Config as cms

# ttbar semi muonique
b2gSingleMuonHLTValidation = cms.EDAnalyzer('B2GSingleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/B2GHLTValidation/B2G/SemiMuonic/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(30.),
        etaElectrons = cms.untracked.double(2.5),
        minElectrons = cms.untracked.uint32(0),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(40.),
        etaMuons     = cms.untracked.double(2.1),
        minMuons     = cms.untracked.uint32(1),
        # Jets
        sJets        = cms.untracked.string('ak4PFJetsCHS'),
        minJets      = cms.untracked.uint32(0),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Mu40_eta2p1']),
)

# ttbar semi electronique
b2gSingleElectronHLTValidation = cms.EDAnalyzer('B2GSingleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/B2GHLTValidation/B2G/SemiElectronic/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(30.),
        etaElectrons = cms.untracked.double(2.5),
        minElectrons = cms.untracked.uint32(1),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(26.),
        etaMuons     = cms.untracked.double(2.1),
        minMuons     = cms.untracked.uint32(0),
        # Jets
        sJets        = cms.untracked.string('ak4PFJetsCHS'),
        ptJets0      = cms.untracked.double(150.),
        ptJets1      = cms.untracked.double(25.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet150_PFNoPUJet25']),
)
