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

#ATHER
topEfficiency_singleMu = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/Top/SingleLepton/SingleMuon/"),
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
        "effic_jetPt_2       'efficiency vs sub-leading jet pt; jet pt [GeV]; efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'efficiency vs sub-leading jet eta; jet eta ; efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'efficiency vs sub-leading jet phi; jet phi ; efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_numerator       jetEtaPhi_denominator",
    ),
)


topEfficiency_diElec = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/Top/DiLepton/DiElectron/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
                                      
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metME       'efficiency vs MET; MET [GeV]; efficiency' metME_numerator       metME_denominator",
        "effic_elePt_1       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_2       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_2_numerator       elePt_2_denominator",
        "effic_eleEta_2       'efficiency vs electron eta; electron eta ; efficiency' eleEta_2_numerator       eleEta_2_denominator",
        "effic_elePhi_2       'efficiency vs electron phi; electron phi ; efficiency' elePhi_2_numerator       elePhi_2_denominator",
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



topEfficiency_diMu = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/Top/DiLepton/DiMuon/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
                                      
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metME       'efficiency vs MET; MET [GeV]; efficiency' metME_numerator       metME_denominator",
        "effic_muPt_1       'efficiency vs muctron pt; muctron pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muctron eta; muctron eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muctron phi; muctron phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_2       'efficiency vs muctron pt; muctron pt [GeV]; efficiency' muPt_2_numerator       muPt_2_denominator",
        "effic_muEta_2       'efficiency vs muctron eta; muctron eta ; efficiency' muEta_2_numerator       muEta_2_denominator",
        "effic_muPhi_2       'efficiency vs muctron phi; muctron phi ; efficiency' muPhi_2_numerator       muPhi_2_denominator",
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


topEfficiency_ElecMu = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/Top/DiLepton/ElecMuon/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
                                      
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metME       'efficiency vs MET; MET [GeV]; efficiency' metME_numerator       metME_denominator",
        "effic_muPt_1       'efficiency vs muctron pt; muctron pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muctron eta; muctron eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muctron phi; muctron phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
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


singleTopEfficiency_singleMu = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/SingleTop/SingleMuon/"),
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
    + topEfficiency_singleMu
    + topEfficiency_diElec
    + topEfficiency_diMu
    + topEfficiency_ElecMu
    + singleTopEfficiency_singleMu
)
