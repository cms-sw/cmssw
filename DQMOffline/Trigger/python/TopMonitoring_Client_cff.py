import FWCore.ParameterSet.Config as cms

topEfficiency_elejets = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/EleJet/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metME       'efficiency vs MET; MET [GeV]; efficiency' metME_numerator       metME_denominator",
        "effic_elePt_1       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_jetPt_1       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet phi; jet phi ; efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_jetPt_2       'efficiency vs sub-leading jet pt; jet pt [GeV]; efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'efficiency vs sub-leading jet eta; jet eta ; efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'efficiency vs sub-leading jet phi; jet phi ; efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_numerator       jetEtaPhi_denominator",
    ),
)

topEfficiency_eleHT = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/EleHT/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metME       'efficiency vs MET; MET [GeV]; efficiency' metME_numerator       metME_denominator",
        "effic_elePt_1       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_jetPt_1       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet phi; jet phi ; efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_jetPt_2       'efficiency vs sub-leading jet pt; jet pt [GeV]; efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'efficiency vs sub-leading jet eta; jet eta ; efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'efficiency vs sub-leading jet phi; jet phi ; efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_numerator       jetEtaPhi_denominator",
    ),
)

topClient = cms.Sequence(
    topEfficiency_elejets
    + topEfficiency_eleHT
)
