import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

metEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/MET/*"),
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

metClient = cms.Sequence(
    metEfficiency
)

