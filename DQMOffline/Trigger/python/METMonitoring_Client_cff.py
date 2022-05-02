import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

metEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/MET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met            'MET turnON;              PF MET [GeV];               efficiency'         met_numerator                  met_denominator",
        "effic_met_variable   'MET turnON;              PF MET [GeV];               efficiency'         met_variable_numerator         met_variable_denominator",
        "effic_metPhi         'MET efficiency vs phi;   PF MET phi [rad];           efficiency'         metPhi_numerator               metPhi_denominator",
        "effic_ht             'HT turnON;               PF HT [GeV];                efficiency'         ht_numerator                   ht_denominator",
        "effic_ht_variable    'HT turnON;               PF HT [GeV];                efficiency'         ht_variable_numerator          ht_variable_denominator",
        "effic_deltaphimetj1  'DELTAPHI turnON;         DELTA PHI (PFMET, PFJET1);  efficiency'         deltaphi_metjet1_numerator     deltaphi_metjet1_denominator",
        "effic_deltaphij1j2   'DELTAPHI turnON;         DELTA PHI (PFJET1, PFJET2); efficiency'         deltaphi_jet1jet2_numerator    deltaphi_jet1jet2_denominator",
        #==============================================================================================================================================
        "effic_jetPhi_1       'JET_PHI_1;               LEADING JET #phi;           efficiency'         jetPhi_1_numerator             jetPhi_1_denominator",
        "effic_jetPhi_2       'JET_PHI_2;               SUBLEADING JET #phi;        efficiency'         jetPhi_2_numerator             jetPhi_2_denominator",
        "effic_jetEta_1       'JET_ETA1;                LEADING JET #eta;           efficiency'         jetEta_1_numerator             jetEta_1_denominator",
        "effic_jetEta_2       'JET_ETA2;                SUBLEADING JET #eta;        efficiency'         jetEta_2_numerator             jetEta_2_denominator",
        "effic_jetPt_1        'JET_PT1 turnON;          LEADING JET PT [GeV];       efficiency'         jetPt_1_numerator              jetPt_1_denominator",
        "effic_jetPt_2        'JET_PT2 turnON;          SUBLEADING JET PT [GeV];    efficiency'         jetPt_2_numerator              jetPt_2_denominator",
        "effic_NJets          'nJets;                   nJets;                      efficiency'         NJets_numerator                NJets_denominator",
        "effic_jetPhi_inlc    'JET_PHI;                 JET #phi;                   efficiency'         jetPhi_numerator               jetPhi_denominator",
        "effic_jetEta_incl    'JET_ETA;                 JET #eta;                   efficiency'         jetEta_numerator               jetEta_denominator",
        "effic_jetPt_incl     'JET_PT turnON;           JET PT [GeV];               efficiency'         jetPt_numerator                jetPt_denominator"
     
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency'  metVsLS_numerator  metVsLS_denominator",
        "effic_ht_vs_LS  'HT efficiency vs LS;  LS; PF HT efficiency'   htVsLS_numerator   htVsLS_denominator"
    ),
  
)

caloMHTEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/CaloMHT/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met            'MET turnON;              PF MET [GeV];               efficiency'         met_numerator                  met_denominator",
        "effic_met_variable   'MET turnON;              PF MET [GeV];               efficiency'         met_variable_numerator         met_variable_denominator",
        "effic_metPhi         'MET efficiency vs phi;   PF MET phi [rad];           efficiency'         metPhi_numerator               metPhi_denominator",
        "effic_ht             'HT turnON;               PF HT [GeV];                efficiency'         ht_numerator                   ht_denominator",
        "effic_ht_variable    'HT turnON;               PF HT [GeV];                efficiency'         ht_variable_numerator          ht_variable_denominator",
        "effic_deltaphimetj1  'DELTAPHI turnON;         DELTA PHI (PFMET, PFJET1);  efficiency'         deltaphi_metjet1_numerator     deltaphi_metjet1_denominator",
        "effic_deltaphij1j2   'DELTAPHI turnON;         DELTA PHI (PFJET1, PFJET2); efficiency'         deltaphi_jet1jet2_numerator    deltaphi_jet1jet2_denominator",
        #==============================================================================================================================================
        "effic_jetPhi_1       'JET_PHI_1;               LEADING JET #phi;           efficiency'         jetPhi_1_numerator             jetPhi_1_denominator",
        "effic_jetPhi_2       'JET_PHI_2;               SUBLEADING JET #phi;        efficiency'         jetPhi_2_numerator             jetPhi_2_denominator",
        "effic_jetEta_1       'JET_ETA1;                LEADING JET #eta;           efficiency'         jetEta_1_numerator             jetEta_1_denominator",
        "effic_jetEta_2       'JET_ETA2;                SUBLEADING JET #eta;        efficiency'         jetEta_2_numerator             jetEta_2_denominator",
        "effic_jetPt_1        'JET_PT1 turnON;          LEADING JET PT [GeV];       efficiency'         jetPt_1_numerator              jetPt_1_denominator",
        "effic_jetPt_2        'JET_PT2 turnON;          SUBLEADING JET PT [GeV];    efficiency'         jetPt_2_numerator              jetPt_2_denominator",
        "effic_NJets          'nJets;                   nJets;                      efficiency'         NJets_numerator                NJets_denominator",
        "effic_jetPhi_inlc    'JET_PHI;                 JET #phi;                   efficiency'         jetPhi_numerator               jetPhi_denominator",
        "effic_jetEta_incl    'JET_ETA;                 JET #eta;                   efficiency'         jetEta_numerator               jetEta_denominator",
        "effic_jetPt_incl     'JET_PT turnON;           JET PT [GeV];               efficiency'         jetPt_numerator                jetPt_denominator"

    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency'  metVsLS_numerator  metVsLS_denominator",
        "effic_ht_vs_LS  'HT efficiency vs LS;  LS; PF HT efficiency'   htVsLS_numerator   htVsLS_denominator"
    ),

)

metClient = cms.Sequence(
    metEfficiency
    +caloMHTEfficiency
)

