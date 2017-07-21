import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

BTVEfficiency_BTagMu_DiJet = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/BTV/BTagMu_DiJet/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metME       'efficiency vs MET; MET [GeV]; efficiency' metME_numerator       metME_denominator",
        "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_jetPt_1       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet phi; jet phi ; efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_jetPt_2       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'efficiency vs leading jet phi; jet phi ; efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_jetPt_1_variableBinning       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetEta_1_variableBinning       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_eventHT_variableBinning       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_muMulti       'efficiency vs muon multiplicity; muon multiplicity; efficiency' muMulti_numerator       muMulti_denominator",
        "effic_eleMulti       'efficiency vs electron multiplicity; electron multiplicity; efficiency' eleMulti_numerator       eleMulti_denominator",
        "effic_muPtEta_1       'efficiency vs muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator       muPtEta_1_denominator",
        "effic_muEtaPhi_1       'efficiency vs muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator       muEtaPhi_1_denominator",
        "effic_jetPtEta_1       'efficiency vs jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetPtEta_2       'efficiency vs jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetEtaPhi_2       'efficiency vs jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_muPt_jetPt       'efficiency vs muon pt - jet pt; muon pt [GeV] ; jet pt [GeV]' muPt_jetPt_numerator       muPt_jetPt_denominator",
        "effic_muPt_eventHT       'efficiency vs muon pt - event HT; muon pt [GeV] ; event HT [GeV]' muPt_eventHT_numerator       muPt_eventHT_denominator",
        "effic_bjetPt_1      'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet eta; bjet eta ; efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet eta; bjet eta ; efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_DeltaR_jet_Mu    'efficiency vs #DeltaR between jet and mu; #DeltaR(jet,mu) ; efficiency' DeltaR_jet_Mu_numerator  DeltaR_jet_Mu_denominator",

    ),
)

BTVEfficiency_BTagMu_Jet = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/BTV/BTagMu_Jet/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metME       'efficiency vs MET; MET [GeV]; efficiency' metME_numerator       metME_denominator",
        "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_jetPt_1       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet phi; jet phi ; efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_jetPt_1_variableBinning       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetEta_1_variableBinning       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_eventHT_variableBinning       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_muMulti       'efficiency vs muon multiplicity; muon multiplicity; efficiency' muMulti_numerator       muMulti_denominator",
        "effic_eleMulti       'efficiency vs electron multiplicity; electron multiplicity; efficiency' eleMulti_numerator       eleMulti_denominator",
        "effic_muPtEta_1       'efficiency vs muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator       muPtEta_1_denominator",
        "effic_muEtaPhi_1       'efficiency vs muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator       muEtaPhi_1_denominator",
        "effic_jetPtEta_1       'efficiency vs jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_muPt_jetPt       'efficiency vs muon pt - jet pt; muon pt [GeV] ; jet pt [GeV]' muPt_jetPt_numerator       muPt_jetPt_denominator",
        "effic_muPt_eventHT       'efficiency vs muon pt - event HT; muon pt [GeV] ; event HT [GeV]' muPt_eventHT_numerator       muPt_eventHT_denominator",
        "effic_bjetPt_1      'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet eta; bjet eta ; efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet eta; bjet eta ; efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_DeltaR_jet_Mu    'efficiency vs #DeltaR between jet and mu; #DeltaR(jet,mu) ; efficiency' DeltaR_jet_Mu_numerator  DeltaR_jet_Mu_denominator",

    ),
)

BTVEfficiency_PFJet = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/BTV/PFJet/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metME       'efficiency vs MET; MET [GeV]; efficiency' metME_numerator       metME_denominator",
        "effic_jetPt_1       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet phi; jet phi ; efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_jetPt_1_variableBinning       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetEta_1_variableBinning       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_eventHT_variableBinning       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_muMulti       'efficiency vs muon multiplicity; muon multiplicity; efficiency' muMulti_numerator       muMulti_denominator",
        "effic_eleMulti       'efficiency vs electron multiplicity; electron multiplicity; efficiency' eleMulti_numerator       eleMulti_denominator",
        "effic_jetPtEta_1       'efficiency vs jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_bjetPt_1      'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet eta; bjet eta ; efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet eta; bjet eta ; efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_DeltaR_jet_Mu    'efficiency vs #DeltaR between jet and mu; #DeltaR(jet,mu) ; efficiency' DeltaR_jet_Mu_numerator  DeltaR_jet_Mu_denominator",

    ),
)


btaggingClient = cms.Sequence(

    BTVEfficiency_BTagMu_DiJet
    + BTVEfficiency_BTagMu_Jet
    + BTVEfficiency_PFJet

)

