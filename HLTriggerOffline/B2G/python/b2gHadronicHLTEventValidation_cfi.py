import FWCore.ParameterSet.Config as cms

# single jet validation
b2gSingleJetHLTValidation = cms.EDAnalyzer('B2GHadronicHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/B2GHLTValidation/B2G/SingleJet/'),
        # Jets
        sJets        = cms.untracked.string('ak8PFJetsCHS'),
        ptJets0      = cms.untracked.double(400.),
        etaJets      = cms.untracked.double(2.4),
        minJets      = cms.untracked.uint32(1),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV0p45',
                                              'HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV0p45',
                                              'HLT_AK8PFJet360_TrimMass30',
                                              'HLT_AK8PFHT700_TrimR0p1PT0p03Mass50',
                                              'HLT_AK8PFHT650_TrimR0p1PT0p03Mass50',
                                              'HLT_AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV0p45']),
)

b2gDiJetHLTValidation = cms.EDAnalyzer('B2GHadronicHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/B2GHLTValidation/B2G/DiJet/'),
        # Jets
        sJets        = cms.untracked.string('ak8PFJetsCHS'),
        ptJets0      = cms.untracked.double(200.),
        ptJets1      = cms.untracked.double(200.),
        etaJets      = cms.untracked.double(2.4),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV0p45',
                                              'HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV0p45',
                                              'HLT_AK8PFJet360_TrimMass30',
                                              'HLT_AK8PFHT700_TrimR0p1PT0p03Mass50',
                                              'HLT_AK8PFHT650_TrimR0p1PT0p03Mass50',
                                              'HLT_AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV0p45']),
)
