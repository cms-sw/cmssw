import FWCore.ParameterSet.Config as cms

b2gjetEfficiency = cms.EDProducer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/B2G/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                                                                    
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_pfjetpT          'Jet pT turnON;            PFJet(pT) [GeV]; efficiency'     pfjetpT_numerator          pfjetpT_denominator",
        "effic_pfjetpT_variable 'Jet pT turnON;            PFJet(pT) [GeV]; efficiency'     pfjetpT_variable_numerator pfjetpT_variable_denominator",
        "effic_pfjetPhi         'Jet efficiency vs #phi; PF Jet #phi [rad]; efficiency'     pfjetPhi_numerator         pfjetPhi_denominator",
        "effic_pfjetEta         'Jet efficiency vs #eta; PF Jet #eta [rad]; efficiency'     pfjetEta_numerator         pfjetEta_denominator",
        "effic_ht          'HT turnON;                                PF HT [GeV]; efficiency'     ht_numerator          ht_denominator",
        "effic_ht_variable 'HT turnON;                                PF HT [GeV]; efficiency'     ht_variable_numerator ht_variable_denominator",
        "effic_mjj_variable 'Mjj turnON;            invariant dijetmass Mjj [GeV]; efficiency'     mjj_variable_numerator mjj_variable_denominator",
        "effic_softdrop_variable 'Softdrop turnON; leading jet softdropmass [GeV]; efficiency'     softdrop_variable_numerator softdrop_variable_denominator",
        "effic_deltaphimetj1          'DELTAPHI turnON;            DELTA PHI (PFMET, PFJET1); efficiency'     deltaphimetj1_numerator          deltaphimetj1_denominator",
        "effic_deltaphij1j2          'DELTAPHI turnON;            DELTA PHI (PFJET1, PFJET2); efficiency'     deltaphij1j2_numerator          deltaphij1j2_denominator",
    ),

    efficiencyProfile = cms.untracked.vstring(
        "effic_pfjetpT_vs_LS 'JET efficiency vs LS; LS; PF JET efficiency' jetpTVsLS_numerator jetpTVsLS_denominator",
        "effic_ht_vs_LS 'HT efficiency vs LS; LS; PF HT efficiency' htVsLS_numerator htVsLS_denominator", 
    ),
)

b2gClient = cms.Sequence(
    b2gjetEfficiency
)

