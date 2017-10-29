#Soureek updated HLT_Ele23_WPLoose_Gsf_CentralPFJet30_BTagCVS07_v to HLT_Ele23_WPLoose_Gsf_CentralPFJet30_BTagCSV07_v
import FWCore.ParameterSet.Config as cms

# single top muonique
SingleTopSingleMuonHLTValidation = cms.EDAnalyzer('TopSingleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/TopHLTValidation/SingleTop/SingleMuon/'),
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
        ptJets       = cms.untracked.double(40.),
        etaJets      = cms.untracked.double(5.),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        iTrigger     = cms.untracked.InputTag("TriggerResults","","HLT"),

### Updating to HLT paths to be monitored by TOP PAG in 2017
        vsPaths     = cms.untracked.vstring(['HLT_Mu20_v*',
                                             'HLT_TkMu20_v*' ,
                                             'HLT_IsoMu20_v*',
                                             'HLT_IsoTkMu20_v*',
                                             'HLT_IsoMu24_eta2p1_v*',
                                             'HLT_IsoMu24_v*',
                                             'HLT_IsoTkMu24_eta2p1_v*',
                                             'HLT_IsoTkMu24_v*',
                                             'HLT_Mu27_v*',
                                             'HLT_TkMu27_v*',
                                             'HLT_IsoMu27_v*',
                                             'HLT_IsoTkMu27_v*',
                                             'HLT_TkMu50_v*',
                                             'HLT_Mu50_v*']),
)

# single top electronique
SingleTopSingleElectronHLTValidation = cms.EDAnalyzer('TopSingleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/TopHLTValidation/SingleTop/SingleElectron/'),
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
        ptJets       = cms.untracked.double(40.),
        etaJets      = cms.untracked.double(5.),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        iTrigger     = cms.untracked.InputTag("TriggerResults","","HLT"),

### Updating to HLT paths to be monitored by TOP PAG in 2017                                                                                                                 
        vsPaths     = cms.untracked.vstring(['HLT_Ele30_eta2p1_WPTight_Gsf_v*',
                                             'HLT_Ele35_WPTight_Gsf_v*',
                                             'HLT_Ele38_WPTight_Gsf_v*',
                                             'HLT_Ele40_WPTight_Gsf_v*']),
)
