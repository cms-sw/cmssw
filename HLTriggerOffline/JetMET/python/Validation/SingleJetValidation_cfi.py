import FWCore.ParameterSet.Config as cms

rootfile="ttbar_output.root"

# calojetcoll="iterativeCone5CaloJets"
#calojetcoll="hltIterativeCone5CaloJets"
#calojetcoll="hltAntiKT5L2L3CorrCaloJets"
calojetcoll="hltAntiKT5PFJets"

#hltlow15  ="HLT_L1SingleJet36"
#hltname15="HLT_Jet30"
#folderjet15="HLT/HLTJETMET/SingleJet30"
foldernm="HLT/HLTJETMET/"

SingleJetMetPaths = cms.EDAnalyzer("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    DQMFolder             = cms.untracked.string(foldernm),
    PatternJetTrg             = cms.untracked.string("HLT_(PF)?Jet([0-9])+(U)?(_v[0-9]+)?$"),
    PatternMetTrg             = cms.untracked.string("HLT_(PF)?M([E,H])+T([0-9])+(_v[0-9]+)?$"),
    PatternMuTrg             = cms.untracked.string("HLT_Mu([0-9])+(_v[0-9]+)?$"),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
    #HLTLow                = cms.untracked.InputTag(hltlow15),
    #HLTPath               = cms.untracked.InputTag(hltname15),
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("ak5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetCalo"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),
    #WriteFile = cms.untracked.bool(True)                               
)

SingleJetValidation = cms.Sequence(SingleJetMetPaths)
