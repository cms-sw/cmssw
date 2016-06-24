import FWCore.ParameterSet.Config as cms

# ttbar semi muonique
topSingleMuonHLTValidation = cms.EDAnalyzer('TopSingleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/TopHLTValidation/Top/SemiMuonic/'),
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
        sJets        = cms.untracked.string('ak4PFJetsCHS'),
        ptJets       = cms.untracked.double(20.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(4),
        # Trigger
        iTrigger     = cms.untracked.InputTag("TriggerResults","","HLT"),

### Updating to HLT paths to be monitored by TOP PAG in 2016                                                                                                                 
        vsPaths     = cms.untracked.vstring(['HLT_IsoMu18_v', 'HLT_IsoMu20_v', 'HLT_IsoMu22_v', 'HLT_IsoMu24_v', 'HLT_IsoTkMu18_v', 'HLT_IsoTkMu20_v', 'HLT_IsoTkMu22_v', 'HLT_IsoTkMu24_v','HLT_IsoMu17_eta2p1_v','HLT_IsoMu20_eta2p1_v','HLT_IsoMu24_eta2p1_v','HLT_IsoTkMu20_eta2p1_v','HLT_IsoTkMu24_eta2p1_v3']),
)

# ttbar semi electronique
topSingleElectronHLTValidation = cms.EDAnalyzer('TopSingleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/TopHLTValidation/Top/SemiElectronic/'),
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
        sJets        = cms.untracked.string('ak4PFJetsCHS'),
        ptJets       = cms.untracked.double(20.),
        etaJets      = cms.untracked.double(2.5),
        minJets      = cms.untracked.uint32(4),
        # Trigger
        iTrigger     = cms.untracked.InputTag("TriggerResults","","HLT"),

### Updating to HLT paths to be monitored by TOP PAG in 2016
        vsPaths     = cms.untracked.vstring(['HLT_Ele27_WPLoose_Gsf_v','HLT_Ele25_WPTight_Gsf_v','HLT_Ele23_WPLoose_Gsf_v','HLT_Ele25_eta2p1_WPTight_Gsf_v', 'HLT_Ele27_WPTight_Gsf_v','HLT_Ele27_eta2p1_WPLoose_Gsf_v', 'HLT_Ele22_eta2p1_WPLoose_Gsf_v', 'HLT_Ele24_eta2p1_WPLoose_Gsf_v','HLT_Ele25_eta2p1_WPLoose_Gsf_v']),
)
