import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

metEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/MET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met          'MET turnON;            PF MET [GeV]; efficiency'     met_numerator          met_denominator",
        "effic_met_variable 'MET turnON;            PF MET [GeV]; efficiency'     met_variable_numerator met_variable_denominator",
        "effic_metPhi       'MET efficiency vs phi; PF MET phi [rad]; efficiency' metPhi_numerator       metPhi_denominator",
        "effic_ht          'HT turnON;            PF HT [GeV]; efficiency'     ht_numerator          ht_denominator",
        "effic_ht_variable 'HT turnON;            PF HT [GeV]; efficiency'     ht_variable_numerator ht_variable_denominator",
        "effic_deltaphimetj1          'DELTAPHI turnON;            DELTA PHI (PFMET, PFJET1); efficiency'     deltaphimetj1_numerator          deltaphimetj1_denominator",
        "effic_deltaphij1j2          'DELTAPHI turnON;            DELTA PHI (PFJET1, PFJET2); efficiency'     deltaphij1j2_numerator          deltaphij1j2_denominator"

    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' metVsLS_numerator metVsLS_denominator",
        "effic_ht_vs_LS 'HT efficiency vs LS; LS; PF HT efficiency' htVsLS_numerator htVsLS_denominator"
    ),
  
)

metbtagEfficiency_btag = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/MET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt_1       'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        #
        "effic_jetEta_1      'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_numerator     jetEta_1_denominator",
        #
        "effic_jetPhi_1      'efficiency vs 1st jet phi; jet phi ; efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        #
        "effic_bjetPt_1      'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet eta; bjet eta ; efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        #
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        #
        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        #
        "effic_jetEta_1_variableBinning       'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        #
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        #
        "effic_eventHT_variableBinning       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        #
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        #
        "effic_jetPtEta_1     'efficiency vs 1st jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        #
        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        #
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        #
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        #
        "effic_bjetCSVHT_1 'efficiency vs 1st b-jet csv - event HT; bjet csv ; event HT [GeV]' bjetCSVHT_1_numerator bjetCSVHT_1_denominator"
        ),
)

metClient = cms.Sequence(
    metEfficiency
    + metbtagEfficiency_btag
)

