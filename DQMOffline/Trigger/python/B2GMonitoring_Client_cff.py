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

hltDQMonitorB2G_Client_MuEle = cms.EDProducer("DQMGenericClient",
    subDirs = cms.untracked.vstring("HLT/B2G/Dileptonic/HLT_MuXX_EleXX_CaloIdL_MW"),
    verbose = cms.untracked.uint32(0),
    resolution = cms.vstring(),
    efficiency = cms.vstring(
        "effic_muPt_1   'efficiency vs leading muon p_{T};muon p_{T} [GeV];efficiency' muPt_1_numerator muPt_1_denominator",
        "effic_muEta_1  'efficiency vs leading muon #eta;muon #eta;efficiency' muEta_1_numerator muEta_1_denominator",
        "effic_muPhi_1  'efficiency vs leading muon #phi;muon #phi;efficiency' muPhi_1_numerator muPhi_1_denominator",
        "effic_muMulti  'efficiency vs muon multiplicity;muon multiplicity;efficiency' muMulti_numerator muMulti_denominator",
        "effic_elePt_1  'efficiency vs electron p_{T};electron p_{T} [GeV];efficiency' elePt_1_numerator elePt_1_denominator",
        "effic_eleEta_1 'efficiency vs electron #eta;electron #eta;efficiency' eleEta_1_numerator eleEta_1_denominator",
        "effic_elePhi_1 'efficiency vs electron #phi;electron #phi;efficiency' elePhi_1_numerator elePhi_1_denominator",
        "effic_eleMulti 'efficiency vs electron multiplicity;electron multiplicity;efficiency' eleMulti_numerator eleMulti_denominator",
    ),
)

hltDQMonitorB2G_Client_MuTkMu = cms.EDProducer("DQMGenericClient",
    subDirs = cms.untracked.vstring("HLT/B2G/Dileptonic/HLT_Mu37_TkMu27"),
    verbose = cms.untracked.uint32(0),
    resolution = cms.vstring(),
    efficiency = cms.vstring(
        "effic_muPt_1  'efficiency vs leading muon p_{T};muon p_{T} [GeV];efficiency' muPt_1_numerator muPt_1_denominator",
        "effic_muEta_1 'efficiency vs leading muon #eta;muon #eta;efficiency' muEta_1_numerator muEta_1_denominator",
        "effic_muPhi_1 'efficiency vs leading muon #phi;muon #phi;efficiency' muPhi_1_numerator muPhi_1_denominator",
        "effic_muPt_2  'efficiency vs sub-leading muon p_{T};muon p_{T} [GeV];efficiency' muPt_2_numerator muPt_2_denominator",
        "effic_muEta_2 'efficiency vs sub-leading muon #eta;muon #eta;efficiency' muEta_2_numerator muEta_2_denominator",
        "effic_muPhi_2 'efficiency vs sub-leading muon #phi;muon #phi;efficiency' muPhi_2_numerator muPhi_2_denominator",
        "effic_muMulti 'efficiency vs muon multiplicity;muon multiplicity;efficiency' muMulti_numerator muMulti_denominator",
    ),
)

b2gClient = cms.Sequence(
    b2gjetEfficiency
  + hltDQMonitorB2G_Client_MuEle
  + hltDQMonitorB2G_Client_MuTkMu
)
