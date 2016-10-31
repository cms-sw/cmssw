import FWCore.ParameterSet.Config as cms

# ttbar cross channel
b2gDoubleLeptonEleMuHLTValidation = cms.EDAnalyzer('B2GDoubleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/B2GHLTValidation/B2G/EleMu/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(25.),
        etaElectrons = cms.untracked.double(2.5),
        minElectrons = cms.untracked.uint32(1),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(25.),
        etaMuons     = cms.untracked.double(2.4),
        minMuons     = cms.untracked.uint32(1),

        # Leptons
        minLeptons = cms.untracked.uint32(2),

        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_Mu37_Ele27_CaloIdL_GsfTrkIdVL','HLT_Mu27_Ele37_CaloIdL_GsfTrkIdVL']),
)

# ttbar double electron
b2gDoubleElectronHLTValidation = cms.EDAnalyzer('B2GDoubleLeptonHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/B2GHLTValidation/B2G/DoubleEle/'),
        # Electrons
        sElectrons   = cms.untracked.string('gedGsfElectrons'),
        ptElectrons  = cms.untracked.double(25.),
        etaElectrons = cms.untracked.double(2.5),
        minElectrons = cms.untracked.uint32(2),
        # Muons
        sMuons       = cms.untracked.string('muons'),
        ptMuons      = cms.untracked.double(40.),
        etaMuons     = cms.untracked.double(2.1),
        minMuons     = cms.untracked.uint32(0),

        # Leptons
        minLeptons = cms.untracked.uint32(2),

        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_DoubleEle37_Ele27_CaloIdL_GsfTrkIdVL']),
)
