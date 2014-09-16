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
        vsPaths      = cms.untracked.vstring(['HLT_IsoMu24_IterTrk02_v', 'HLT_IsoMu24_IterTrk02_TriCentralPFJet60_50_35_v', 'HLT_IsoMu24_IterTrk02_TriCentralPFJet40_v', 'HLT_IsoMu24_IterTrk02_CentralPFJet30_BTagCSV_v']),
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
        vsPaths      = cms.untracked.vstring(['HLT_Ele27_WP85_Gsf_v', 'HLT_Ele27_WP85_Gsf_TriCentralPFJet40_v', 'HLT_Ele27_WP85_Gsf_TriCentralPFJet60_50_35_v', 'HLT_Ele27_WP85_Gsf_CentralPFJet30_BTagCSV_v']),
)
