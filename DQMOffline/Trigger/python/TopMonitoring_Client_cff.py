import FWCore.ParameterSet.Config as cms

topEfficiency = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        # "effic_muPhi       'mu efficiency vs phi; muon phi ; efficiency' muPhi_numerator       muPhi_denominator",
        # "effic_muEta       'mu efficiency vs eta; muon eta ; efficiency' muEta_numerator       muEta_denominator",
        # "effic_muPt       'mu efficiency vs pt; muon pt [GeV]; efficiency' muPt_numerator       muPt_denominator",
        "effic_elePt_1       'electron efficiency vs pt; electron pt [GeV]; efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'electron efficiency vs eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'electron efficiency vs phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_jetPt_1       'jet efficiency vs pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'jet efficiency vs eta; jet eta ; efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'jet efficiency vs phi; jet phi ; efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_jetPt_2       'jet efficiency vs pt; jet pt [GeV]; efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'jet efficiency vs eta; jet eta ; efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'jet efficiency vs phi; jet phi ; efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        # "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' metVsLS_numerator metVsLS_denominator",
        # "effic_jet_vs_LS 'jet efficiency vs LS; LS; jet efficiency' jetVsLS_numerator jetVsLS_denominator",
        # "effic_mu_vs_LS 'mu efficiency vs LS; LS; muon efficiency' muVsLS_numerator muVsLS_denominator",
        # "effic_ele_vs_LS 'ele efficiency vs LS; LS; electron efficiency' eleVsLS_numerator eleVsLS_denominator",
        # "effic_HT_vs_LS 'HT efficiency vs LS; LS; HT efficiency' htVsLS_numerator HTVsLS_denominator",
    ),
)

topClient = cms.Sequence(
    topEfficiency
)
