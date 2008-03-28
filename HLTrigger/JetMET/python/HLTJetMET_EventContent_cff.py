import FWCore.ParameterSet.Config as cms

# Full Event content
HLTJetMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MCJetCorJetIcone5*_*_*', 'keep *_iterativeCone5CaloJets*_*_*', 'keep *_met_*_*', 'keep *_htMet_*_*', 'keep *_htMetIC5_*_*')
)
# RECO content
HLTJetMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MCJetCorJetIcone5*_*_*', 'keep *_iterativeCone5CaloJets*_*_*', 'keep *_met_*_*', 'keep *_htMet_*_*', 'keep *_htMetIC5_*_*')
)
# AOD content
HLTJetMETAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("MCJetCorJetIcone5"), cms.InputTag("iterativeCone5CaloJets"), cms.InputTag("MCJetCorJetIcone5Regional"), cms.InputTag("iterativeCone5CaloJetsRegional"), cms.InputTag("met"), cms.InputTag("htMet"), cms.InputTag("htMetIC5")),
    triggerFilters = cms.VInputTag(cms.InputTag("hlt1jet200"), cms.InputTag("hlt1jet150"), cms.InputTag("hlt1jet110"), cms.InputTag("hlt1jet60"), cms.InputTag("hlt1jet30"), cms.InputTag("hlt2jet150"), cms.InputTag("hlt3jet85"), cms.InputTag("hlt4jet60"), cms.InputTag("hlt1MET65"), cms.InputTag("hlt1MET55"), cms.InputTag("hlt1MET30"), cms.InputTag("hlt1MET20"), cms.InputTag("hlt2jetAco"), cms.InputTag("hlt1jet180"), cms.InputTag("hlt2jet125"), cms.InputTag("hlt3jet60"), cms.InputTag("hlt4jet35"), cms.InputTag("hlt1jet1METAco"), cms.InputTag("hlt2jetvbf"), cms.InputTag("hltnv"), cms.InputTag("hltPhi2metAco"), cms.InputTag("hltPhiJet1metAco"), cms.InputTag("hltPhiJet2metAco"), cms.InputTag("hltPhiJet1Jet2Aco"), cms.InputTag("hlt1SumET120"), cms.InputTag("hlt1HT400"), cms.InputTag("hlt1HT350"), cms.InputTag("hltRapGap"), cms.InputTag("hltdijetave110"), cms.InputTag("hltdijetave150"), cms.InputTag("hltdijetave200"), cms.InputTag("hltdijetave30"), cms.InputTag("hltdijetave60")),
    outputCommands = cms.untracked.vstring()
)

