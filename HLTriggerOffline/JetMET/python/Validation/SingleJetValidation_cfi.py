import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SingleJetMetPaths = DQMEDAnalyzer('HLTJetMETValidation',
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    DQMFolder             = cms.untracked.string("HLT/HLTJETMET/"),
    PatternJetTrg         = cms.untracked.string("HLT_PF(NoPU)?Jet([0-9])+(_v[0-9]+)?$"),                                   
    PatternMetTrg         = cms.untracked.string("HLT_+[Calo|PF]+MET([0-9])+[_NotCleaned|_BeamHaloCleaned]+(_v[0-9]+)?$"),
    PatternMuTrg          = cms.untracked.string("HLT_Mu([0-9])+(_v[0-9]+)?$"),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
    PFJetAlgorithm        = cms.untracked.InputTag("hltAK4PFJets"),
    GenJetAlgorithm       = cms.untracked.InputTag("ak4GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetCalo"),
    HLTriggerResults      = cms.InputTag("TriggerResults::HLT"),
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(SingleJetMetPaths,
                       PatternJetTrg = "HLT_(AK4)?PFPuppiJet([0-9])+(_v[0-9]+)?$",
                       PFJetAlgorithm = "hltAK4PFPuppiJetsCorrected"
                       )

SingleJetValidation = cms.Sequence(SingleJetMetPaths)
