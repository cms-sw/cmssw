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
        vsPaths      = cms.untracked.vstring(['HLT_IsoMu18_CentralPFJet30_BTagCSV07_v', 'HLT_IsoMu18_TriCentralPFJet50_40_30_v', 'HLT_IsoMu22_v', 'HLT_IsoMu18_v', 'HLT_IsoMu22_TriCentralPFJet50_40_30_v', 'HLT_IsoMu22_CentralPFJet30_BTagCSV07_v', 'HLT_IsoMu20_eta2p1_TriCentralPFJet30_v', 'HLT_IsoMu20_eta2p1_TriCentralPFJet50_40_30_v', 'HLT_IsoMu20_eta2p1_CentralPFJet30_BTagCSV07_v', 'HLT_IsoMu20_eta2p1_v', 'HLT_IsoMu24_eta2p1_TriCentralPFJet30_v', 'HLT_IsoMu24_eta2p1_TriCentralPFJet50_40_30_v', 'HLT_IsoMu24_eta2p1_CentralPFJet30_BTagCSV07_v']),
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

## Soureek changing path name HLT_Ele23_WPLoose_Gsf_CentralPFJet30_BTagCVS07_v to HLT_Ele23_WPLoose_Gsf_CentralPFJet30_BTagCSV07_v
		vsPaths      = cms.untracked.vstring(['HLT_Ele23_WPLoose_Gsf_TriCentralPFJet50_40_30_v', 'HLT_Ele23_WPLoose_Gsf_CentralPFJet30_BTagCSV07_v', 'HLT_Ele27_WPLoose_Gsf_WHbbBoost_v', 'HLT_Ele27_WPLoose_Gsf_v', 'HLT_Ele27_WPLoose_Gsf_CentralPFJet30_BTagCSV07_v' 'HLT_Ele27_WPLoose_Gsf_TriCentralPFJet50_40_30_v', 'HLT_Ele27_eta2p1_WPLoose_Gsf_TriCentralPFJet30_v', 'HLT_Ele27_eta2p1_WPLoose_Gsf_TriCentralPFJet50_40_30_v', 'HLT_Ele27_eta2p1_WPLoose_Gsf_v', 'HLT_Ele27_eta2p1_WPLoose_Gsf_CentralPFJet30_BTagCSV07_v', 'HLT_Ele32_eta2p1_WPLoose_Gsf_TriCentralPFJet30_v', 'HLT_Ele32_eta2p1_WPLoose_Gsf_TriCentralPFJet50_40_30_v', 'HLT_Ele32_eta2p1_WPLoose_Gsf_v', 'HLT_Ele32_eta2p1_WPLoose_Gsf_CentralPFJet30_BTagCSV07_v']),
)
