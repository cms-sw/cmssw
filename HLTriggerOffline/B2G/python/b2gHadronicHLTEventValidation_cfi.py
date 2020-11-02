import FWCore.ParameterSet.Config as cms

# single jet validation
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
b2gSingleJetHLTValidation = DQMEDAnalyzer('B2GHadronicHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/B2GHLTValidation/B2G/SingleJet/'),
        # Jets
        sJets        = cms.untracked.string('ak8PFJetsPuppi'),
        ptJets0      = cms.untracked.double(400.),
        etaJets      = cms.untracked.double(2.4),
        minJets      = cms.untracked.uint32(1),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20',
                                              'HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV_p20',
                                              'HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p087',
                                              'HLT_AK8DiPFJet300_200_TrimMass30_BTagCSV_p20',
                                              'HLT_AK8DiPFJet300_200_TrimMass30_BTagCSV_p087',
                                              'HLT_AK8PFJet360_TrimMass30',
                                              'HLT_AK8PFJet400_TrimMass30',
                                              'HLT_AK8PFHT800_TrimMass50',
                                              'HLT_AK8PFHT750_TrimMass50',
                                              'HLT_AK8PFHT700_TrimR0p1PT0p03Mass50',
                                              'HLT_AK8PFHT650_TrimR0p1PT0p03Mass50',
                                              'HLT_AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV_p20']),
)

b2gDiJetHLTValidation = DQMEDAnalyzer('B2GHadronicHLTValidation',
        # Directory
        sDir         = cms.untracked.string('HLT/B2GHLTValidation/B2G/DiJet/'),
        # Jets
        sJets        = cms.untracked.string('ak8PFJetsPuppi'),
        ptJets0      = cms.untracked.double(200.),
        ptJets1      = cms.untracked.double(200.),
        etaJets      = cms.untracked.double(2.4),
        minJets      = cms.untracked.uint32(2),
        # Trigger
        sTrigger     = cms.untracked.string("TriggerResults"),
        vsPaths      = cms.untracked.vstring(['HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20',
                                              'HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV_p20',
                                              'HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p087',
                                              'HLT_AK8DiPFJet300_200_TrimMass30_BTagCSV_p20',
                                              'HLT_AK8DiPFJet300_200_TrimMass30_BTagCSV_p087',
                                              'HLT_AK8PFJet360_TrimMass30',
                                              'HLT_AK8PFJet400_TrimMass30',
                                              'HLT_AK8PFHT800_TrimMass50',
                                              'HLT_AK8PFHT750_TrimMass50',
                                              'HLT_AK8PFHT700_TrimR0p1PT0p03Mass50',
                                              'HLT_AK8PFHT650_TrimR0p1PT0p03Mass50',
                                              'HLT_AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV_p20']),
)

# puppi jets don't exist in HI wfs, use Cs jets instead
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
(pp_on_AA_2018 | pp_on_PbPb_run3).toModify(b2gSingleJetHLTValidation, sJets = "akCs4PFJets")
(pp_on_AA_2018 | pp_on_PbPb_run3).toModify(b2gDiJetHLTValidation, sJets = "akCs4PFJets")
