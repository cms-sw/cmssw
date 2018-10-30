import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

MssmHbbHLTEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/HIG/MssmHbb/fullhadronic/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(

        "effic_bjetPt_1      'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet eta; bjet eta ; efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet eta; bjet eta ; efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",

        "effic_bjetPt_2      'efficiency vs 2nd b-jet pt; bjet pt [GeV]; efficiency' bjetPt_2_numerator  bjetPt_2_denominator",
        "effic_bjetEta_2     'efficiency vs 2nd b-jet eta; bjet eta ; efficiency'  bjetEta_2_numerator   bjetEta_2_denominator",
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet phi; bjet phi ; efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet csv; bjet CSV; efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",
        "effic_bjetPt_2_variableBinning   'efficiency vs 2nd b-jet pt; bjet pt [GeV]; efficiency' bjetPt_2_variableBinning_numerator   bjetPt_2_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet eta; bjet eta ; efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",
        "effic_bjetEtaPhi_2    'efficiency vs 2nd b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",

        #ADD EFFIC DeltaEtaMax between jet-jet

    ),
)

MssmHbbmuHLTEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/HIG/MssmHbb/semileptonic/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(

        "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_2       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_2_numerator       muPt_2_denominator",
        "effic_muEta_2       'efficiency vs muon eta; muon eta ; efficiency' muEta_2_numerator       muEta_2_denominator",
        "effic_muPhi_2       'efficiency vs muon phi; muon phi ; efficiency' muPhi_2_numerator       muPhi_2_denominator",

        "effic_muPt_1_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPt_2_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_2_variableBinning_numerator       muPt_2_variableBinning_denominator",
        "effic_muEta_2_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_2_variableBinning_numerator       muEta_2_variableBinning_denominator",

        "effic_muMulti       'efficiency vs muon multiplicity; muon multiplicity; efficiency' muMulti_numerator       muMulti_denominator",

        "effic_bjetPt_1      'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet eta; bjet eta ; efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet eta; bjet eta ; efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",

        "effic_bjetPt_2      'efficiency vs 2nd b-jet pt; bjet pt [GeV]; efficiency' bjetPt_2_numerator  bjetPt_2_denominator",
        "effic_bjetEta_2     'efficiency vs 2nd b-jet eta; bjet eta ; efficiency'  bjetEta_2_numerator   bjetEta_2_denominator",
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet phi; bjet phi ; efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet csv; bjet CSV; efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",
        "effic_bjetPt_2_variableBinning   'efficiency vs 2nd b-jet pt; bjet pt [GeV]; efficiency' bjetPt_2_variableBinning_numerator   bjetPt_2_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet eta; bjet eta ; efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",
        "effic_bjetEtaPhi_2    'efficiency vs 2nd b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",

        "effic_DeltaR_jet_Mu    'efficiency vs #DeltaR between jet and mu; #DeltaR(jet,mu) ; efficiency' DeltaR_jet_Mu_numerator  DeltaR_jet_Mu_denominator", # What if bjet?

    ),
)

MssmHbbmuHLTcontrolEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/HIG/MssmHbb/control/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                                                                                                                   
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(

        "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_2       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_2_numerator       muPt_2_denominator",
        "effic_muEta_2       'efficiency vs muon eta; muon eta ; efficiency' muEta_2_numerator       muEta_2_denominator",
        "effic_muPhi_2       'efficiency vs muon phi; muon phi ; efficiency' muPhi_2_numerator       muPhi_2_denominator",

        "effic_muPt_1_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPt_2_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_2_variableBinning_numerator       muPt_2_variableBinning_denominator",
        "effic_muEta_2_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_2_variableBinning_numerator       muEta_2_variableBinning_denominator",

        "effic_muMulti       'efficiency vs muon multiplicity; muon multiplicity; efficiency' muMulti_numerator       muMulti_denominator",

        "effic_jetPt_1      'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator  jetPt_1_denominator",
        "effic_jetEta_1     'efficiency vs 1st jet eta; jet eta ; efficiency'  jetEta_1_numerator   jetEta_1_denominator",
        "effic_jetPhi_1     'efficiency vs 1st jet phi; jet phi ; efficiency'  jetPhi_1_numerator   jetPhi_1_denominator",
        "effic_jetCSV_1     'efficiency vs 1st jet csv; jet CSV; efficiency' jetCSV_1_numerator  jetCSV_1_denominator",
        "effic_jetPt_1_variableBinning   'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator   jetPt_1_variableBinning_denominator",
        "effic_jetEta_1_variableBinning  'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator     jetEta_1_variableBinning_denominator",
        "effic_jetMulti      'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator   jetMulti_denominator",
        "effic_jetPtEta_1    'efficiency vs 1st jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator   jetPtEta_1_denominator",
        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator  jetEtaPhi_1_denominator",

        "effic_DeltaR_jet_Mu    'efficiency vs #DeltaR between jet and mu; #DeltaR(jet,mu) ; efficiency' DeltaR_jet_Mu_numerator  DeltaR_jet_Mu_denominator",

    ),
)

mssmHbbHLTEfficiency = cms.Sequence(

    MssmHbbHLTEfficiency
    + MssmHbbmuHLTEfficiency
    + MssmHbbmuHLTcontrolEfficiency

)
