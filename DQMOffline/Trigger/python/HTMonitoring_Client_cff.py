import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

htEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/HT/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_ht                     'HT turnON;                  PF HT [GeV];                efficiency'        ht_numerator                   ht_denominator",
        "effic_ht_variable            'HT turnON;                  PF HT [GeV];                efficiency'        ht_variable_numerator          ht_variable_denominator",
        "effic_METPhi                 'METPhi turnON;              MET #phi;                   efficiency'        METPhi_numerator               METPhi_denominator",
        "effic_jetPt1                 'JET_PT1 turnON;             LEADING JET PT [GeV];       efficiency'        jetPt1_numerator               jetPt1_denominator",
        "effic_jetPt2                 'JET_PT2 turnON;             SUBLEADING JET PT [GeV];    efficiency'        jetPt2_numerator               jetPt2_denominator",
        "effic_jetEta1                'JET_ETA1 turnON;            LEADING JET #eta;           efficiency'        jetEta1_numerator              jetEta1_denominator",
        "effic_jetEta2                'JET_ETA2 turnON;            SUBLEADING JET #eta;        efficiency'        jetEta2_numerator              jetEta2_denominator",
        "effic_jetPhi1                'JET_PHI1 turnON;            LEADING JET #phi;           efficiency'        jetPhi1_numerator              jetPhi1_denominator",
        "effic_jetPhi2                'JET_PHI2 turnON;            SUBLEADING JET #phi;        efficiency'        jetPhi2_numerator              jetPhi2_denominator",
        "effic_nJets                  'nJets;                      nJets;                      efficiency'        nJets_numerator                nJets_denominator",
        "effic_nJets_HT               'nJets (Pt>30 && eta<2.5);   nJets_;                     efficiency'        nJetsHT_numerator              nJetsHT_denominator"
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_ht_vs_LS               'HT efficiency vs LS;        LS;                         PF HT efficiency'  htVsLS_numerator               htVsLS_denominator"
    ),
  
)

htClient = cms.Sequence(
    htEfficiency
)
