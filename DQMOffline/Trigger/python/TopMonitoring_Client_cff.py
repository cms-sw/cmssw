import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester


topEfficiency_elejets = DQMEDHarvester("DQMGenericClient",
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
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator       elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_jetPt_1_variableBinning       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetEta_1_variableBinning       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_eventHT_variableBinning       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_eleMulti       'efficiency vs electron multiplicity; electron multiplicity; efficiency' eleMulti_numerator       eleMulti_denominator",
        "effic_muMulti       'efficiency vs muon multiplicity; muon multiplicity; efficiency' muMulti_numerator       muMulti_denominator",
        "effic_elePtEta_1       'efficiency vs electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
        "effic_jetPtEta_1       'efficiency vs jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_elePt_jetPt       'efficiency vs electron pt - jet pt; electron pt [GeV] ; jet pt [GeV]' elePt_jetPt_numerator       elePt_jetPt_denominator",
        "effic_elePt_eventHT       'efficiency vs electron pt - event HT; electron pt [GeV] ; event HT [GeV]' elePt_eventHT_numerator       elePt_eventHT_denominator",

    ),
)

topEfficiency_eleHT = DQMEDHarvester("DQMGenericClient",
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
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi for HEP17; jet #eta; jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator       elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_jetPt_1_variableBinning       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetEta_1_variableBinning       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs sub-leading jet pt; jet pt [GeV]; efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs sub-leading jet eta; jet eta ; efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_eventHT_variableBinning       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_eleMulti       'efficiency vs electron multiplicity; electron multiplicity; efficiency' eleMulti_numerator       eleMulti_denominator",
        "effic_muMulti       'efficiency vs muon multiplicity; muon multiplicity; efficiency' muMulti_numerator       muMulti_denominator",
        "effic_elePtEta_1       'efficiency vs electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
        "effic_jetPtEta_1       'efficiency vs jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetPtEta_2       'efficiency vs jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetEtaPhi_2       'efficiency vs jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_elePt_jetPt       'efficiency vs electron pt - jet pt; electron pt [GeV] ; jet pt [GeV]' elePt_jetPt_numerator       elePt_jetPt_denominator",
        "effic_elePt_eventHT       'efficiency vs electron pt - event HT; electron pt [GeV] ; event HT [GeV]' elePt_eventHT_numerator       elePt_eventHT_denominator",
    ),
)

#ATHER
topEfficiency_singleMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/SingleLepton/SingleMuon/*"),
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


topEfficiency_diElec = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/DiLepton/DiElectron/"),
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
        "effic_ele1Pt_ele2Pt    'efficiency vs ele1-ele2 pt; ele1 pt; ele2 pt' ele1Pt_ele2Pt_numerator       ele1Pt_ele2Pt_denominator",
        "effic_ele1Eta_ele2Eta    'efficiency vs ele1-ele2 #eta; ele1 #eta; ele2 #eta' ele1Eta_ele2Eta_numerator       ele1Eta_ele2Eta_denominator",


    ),
)



topEfficiency_diMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/DiLepton/DiMuon/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                      
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metME       'efficiency vs MET; MET [GeV]; efficiency' metME_numerator       metME_denominator",
        "effic_muPt_1       'efficiency vs mu pt; mu pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs mu eta; mu eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs mu phi; mu phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_2       'efficiency vs mu pt; mu pt [GeV]; efficiency' muPt_2_numerator       muPt_2_denominator",
        "effic_muEta_2       'efficiency vs mu eta; mu eta ; efficiency' muEta_2_numerator       muEta_2_denominator",
        "effic_muPhi_2       'efficiency vs mu phi; mu phi ; efficiency' muPhi_2_numerator       muPhi_2_denominator",
        "effic_jetPt_1       'efficiency vs leading jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet eta; jet eta ; efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet phi; jet phi ; efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_jetPt_2       'efficiency vs sub-leading jet pt; jet pt [GeV]; efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'efficiency vs sub-leading jet eta; jet eta ; efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'efficiency vs sub-leading jet phi; jet phi ; efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_numerator       jetEtaPhi_denominator",
        "effic_mu1Pt_mu2Pt    'efficiency vs mu1-mu2 pt; mu1 pt; mu2 pt' mu1Pt_mu2Pt_numerator       mu1Pt_mu2Pt_denominator",
        "effic_mu1Eta_mu2Eta    'efficiency vs mu1-mu2 #eta; mu1 #eta; mu2 #phi' mu1Eta_mu2Eta_numerator      mu1Eta_mu2Eta_denominator",

        
        

    ),
)


topEfficiency_ElecMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/DiLepton/ElecMuon/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metME       'efficiency vs MET; MET [GeV]; efficiency' metME_numerator       metME_denominator",
        "effic_muPt_1       'efficiency vs mu pt; mu pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs mu eta; mu eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs mu phi; mu phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
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
        "effic_elePt_muPt    'efficiency vs ele-mu pt; ele pt; mu pt' elePt_muPt_numerator       elePt_muPt_denominator",
        "effic_eleEta_muEta    'efficiency vs ele-mu #eta; ele #eta; mu #phi' eleEta_muEta_numerator      eleEta_muEta_denominator",
        

    ),
)


# Marina
fullyhadronicEfficiency_Reference = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/FullyHadronic/Reference/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_eventHT_variableBinning       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_variableBinning_numerator    eventHT_variableBinning_denominator",
        ),
)


fullyhadronicEfficiency_DoubleBTag = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/FullyHadronic/DoubleBTag/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt_1       'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetPt_2       'efficiency vs 2nd jet pt; jet pt [GeV]; efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetPt_3       'efficiency vs 3rd jet pt; jet pt [GeV]; efficiency' jetPt_3_numerator       jetPt_3_denominator",
        "effic_jetPt_4       'efficiency vs 4th jet pt; jet pt [GeV]; efficiency' jetPt_4_numerator       jetPt_4_denominator",
        "effic_jetPt_5       'efficiency vs 5th jet pt; jet pt [GeV]; efficiency' jetPt_5_numerator       jetPt_5_denominator",
        "effic_jetPt_6       'efficiency vs 6th jet pt; jet pt [GeV]; efficiency' jetPt_6_numerator       jetPt_6_denominator",
        #
        "effic_jetEta_1      'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_numerator     jetEta_1_denominator",
        "effic_jetEta_2      'efficiency vs 2nd jet eta; jet eta ; efficiency' jetEta_2_numerator     jetEta_2_denominator",
        "effic_jetEta_3      'efficiency vs 3rd jet eta; jet eta ; efficiency' jetEta_3_numerator     jetEta_3_denominator",
        "effic_jetEta_4      'efficiency vs 4th jet eta; jet eta ; efficiency' jetEta_4_numerator     jetEta_4_denominator",
        "effic_jetEta_5      'efficiency vs 5th jet eta; jet eta ; efficiency' jetEta_5_numerator     jetEta_5_denominator",
        "effic_jetEta_6      'efficiency vs 6th jet eta; jet eta ; efficiency' jetEta_6_numerator     jetEta_6_denominator",
        #
        "effic_jetPhi_1      'efficiency vs 1st jet phi; jet phi ; efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        "effic_jetPhi_2      'efficiency vs 2nd jet phi; jet phi ; efficiency'    jetPhi_2_numerator      jetPhi_2_denominator",
        "effic_jetPhi_3      'efficiency vs 3rd jet phi; jet phi ; efficiency'    jetPhi_3_numerator      jetPhi_3_denominator",
        "effic_jetPhi_4      'efficiency vs 4th jet phi; jet phi ; efficiency'    jetPhi_4_numerator      jetPhi_4_denominator",
        "effic_jetPhi_5      'efficiency vs 5th jet phi; jet phi ; efficiency'    jetPhi_5_numerator      jetPhi_5_denominator",
        "effic_jetPhi_6      'efficiency vs 6th jet phi; jet phi ; efficiency'    jetPhi_6_numerator      jetPhi_6_denominator",
        #
        "effic_bjetPt_1      'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetPt_2      'efficiency vs 2nd b-jet pt; bjet pt [GeV]; efficiency' bjetPt_2_numerator  bjetPt_2_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet eta; bjet eta ; efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetEta_2     'efficiency vs 2nd b-jet eta; bjet eta ; efficiency'  bjetEta_2_numerator   bjetEta_2_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet phi; bjet phi ; efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet csv; bjet CSV; efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",
        #
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        #
        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs 2nd jet pt; jet pt [GeV]; efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetPt_3_variableBinning       'efficiency vs 3rd jet pt; jet pt [GeV]; efficiency' jetPt_3_variableBinning_numerator       jetPt_3_variableBinning_denominator",
        "effic_jetPt_4_variableBinning       'efficiency vs 4th jet pt; jet pt [GeV]; efficiency' jetPt_4_variableBinning_numerator       jetPt_4_variableBinning_denominator",
        "effic_jetPt_5_variableBinning       'efficiency vs 5th jet pt; jet pt [GeV]; efficiency' jetPt_5_variableBinning_numerator       jetPt_5_variableBinning_denominator",
        "effic_jetPt_6_variableBinning       'efficiency vs 6th jet pt; jet pt [GeV]; efficiency' jetPt_6_variableBinning_numerator       jetPt_6_variableBinning_denominator",
        #
        "effic_jetEta_1_variableBinning       'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs 2nd jet eta; jet eta ; efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_jetEta_3_variableBinning       'efficiency vs 3rd jet eta; jet eta ; efficiency' jetEta_3_variableBinning_numerator       jetEta_3_variableBinning_denominator",
        "effic_jetEta_4_variableBinning       'efficiency vs 4th jet eta; jet eta ; efficiency' jetEta_4_variableBinning_numerator       jetEta_4_variableBinning_denominator",
        "effic_jetEta_5_variableBinning       'efficiency vs 5th jet eta; jet eta ; efficiency' jetEta_5_variableBinning_numerator       jetEta_5_variableBinning_denominator",
        "effic_jetEta_6_variableBinning       'efficiency vs 6th jet eta; jet eta ; efficiency' jetEta_6_variableBinning_numerator       jetEta_6_variableBinning_denominator",
        #
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetPt_2_variableBinning   'efficiency vs 2nd b-jet pt; bjet pt [GeV]; efficiency' bjetPt_2_variableBinning_numerator   bjetPt_2_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet eta; bjet eta ; efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet eta; bjet eta ; efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",
        #
        "effic_eventHT_variableBinning       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        #
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        #
        "effic_jetPtEta_1     'efficiency vs 1st jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetPtEta_2     'efficiency vs 2nd jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetPtEta_3     'efficiency vs 3rd jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_3_numerator       jetPtEta_3_denominator",
        "effic_jetPtEta_4     'efficiency vs 4th jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_4_numerator       jetPtEta_4_denominator",
        "effic_jetPtEta_5     'efficiency vs 5th jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_5_numerator       jetPtEta_5_denominator",
        "effic_jetPtEta_6     'efficiency vs 6th jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_6_numerator       jetPtEta_6_denominator",
        #
        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetEtaPhi_2    'efficiency vs 2nd jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_3    'efficiency vs 3rd jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_3_numerator       jetEtaPhi_3_denominator",
        "effic_jetEtaPhi_4    'efficiency vs 4th jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_4_numerator       jetEtaPhi_4_denominator",
        "effic_jetEtaPhi_5    'efficiency vs 5th jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_5_numerator       jetEtaPhi_5_denominator",
        "effic_jetEtaPhi_6    'efficiency vs 6th jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_6_numerator       jetEtaPhi_6_denominator",
        #
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",
        #
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_bjetEtaPhi_2    'efficiency vs 2nd b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",
        ),
)



fullyhadronicEfficiency_SingleBTag = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/FullyHadronic/SingleBTag/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt_1       'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetPt_2       'efficiency vs 2nd jet pt; jet pt [GeV]; efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetPt_3       'efficiency vs 3rd jet pt; jet pt [GeV]; efficiency' jetPt_3_numerator       jetPt_3_denominator",
        "effic_jetPt_4       'efficiency vs 4th jet pt; jet pt [GeV]; efficiency' jetPt_4_numerator       jetPt_4_denominator",
        "effic_jetPt_5       'efficiency vs 5th jet pt; jet pt [GeV]; efficiency' jetPt_5_numerator       jetPt_5_denominator",
        "effic_jetPt_6       'efficiency vs 6th jet pt; jet pt [GeV]; efficiency' jetPt_6_numerator       jetPt_6_denominator",
        #
        "effic_jetEta_1      'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_numerator     jetEta_1_denominator",
        "effic_jetEta_2      'efficiency vs 2nd jet eta; jet eta ; efficiency' jetEta_2_numerator     jetEta_2_denominator",
        "effic_jetEta_3      'efficiency vs 3rd jet eta; jet eta ; efficiency' jetEta_3_numerator     jetEta_3_denominator",
        "effic_jetEta_4      'efficiency vs 4th jet eta; jet eta ; efficiency' jetEta_4_numerator     jetEta_4_denominator",
        "effic_jetEta_5      'efficiency vs 5th jet eta; jet eta ; efficiency' jetEta_5_numerator     jetEta_5_denominator",
        "effic_jetEta_6      'efficiency vs 6th jet eta; jet eta ; efficiency' jetEta_6_numerator     jetEta_6_denominator",
        #
        "effic_jetPhi_1      'efficiency vs 1st jet phi; jet phi ; efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        "effic_jetPhi_2      'efficiency vs 2nd jet phi; jet phi ; efficiency'    jetPhi_2_numerator      jetPhi_2_denominator",
        "effic_jetPhi_3      'efficiency vs 3rd jet phi; jet phi ; efficiency'    jetPhi_3_numerator      jetPhi_3_denominator",
        "effic_jetPhi_4      'efficiency vs 4th jet phi; jet phi ; efficiency'    jetPhi_4_numerator      jetPhi_4_denominator",
        "effic_jetPhi_5      'efficiency vs 5th jet phi; jet phi ; efficiency'    jetPhi_5_numerator      jetPhi_5_denominator",
        "effic_jetPhi_6      'efficiency vs 6th jet phi; jet phi ; efficiency'    jetPhi_6_numerator      jetPhi_6_denominator",
        #
        "effic_bjetPt_1      'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetPt_2      'efficiency vs 2nd b-jet pt; bjet pt [GeV]; efficiency' bjetPt_2_numerator  bjetPt_2_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet eta; bjet eta ; efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetEta_2     'efficiency vs 2nd b-jet eta; bjet eta ; efficiency'  bjetEta_2_numerator   bjetEta_2_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet phi; bjet phi ; efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet csv; bjet CSV; efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",
        #
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        #
        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs 2nd jet pt; jet pt [GeV]; efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetPt_3_variableBinning       'efficiency vs 3rd jet pt; jet pt [GeV]; efficiency' jetPt_3_variableBinning_numerator       jetPt_3_variableBinning_denominator",
        "effic_jetPt_4_variableBinning       'efficiency vs 4th jet pt; jet pt [GeV]; efficiency' jetPt_4_variableBinning_numerator       jetPt_4_variableBinning_denominator",
        "effic_jetPt_5_variableBinning       'efficiency vs 5th jet pt; jet pt [GeV]; efficiency' jetPt_5_variableBinning_numerator       jetPt_5_variableBinning_denominator",
        "effic_jetPt_6_variableBinning       'efficiency vs 6th jet pt; jet pt [GeV]; efficiency' jetPt_6_variableBinning_numerator       jetPt_6_variableBinning_denominator",
        #
        "effic_jetEta_1_variableBinning       'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs 2nd jet eta; jet eta ; efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_jetEta_3_variableBinning       'efficiency vs 3rd jet eta; jet eta ; efficiency' jetEta_3_variableBinning_numerator       jetEta_3_variableBinning_denominator",
        "effic_jetEta_4_variableBinning       'efficiency vs 4th jet eta; jet eta ; efficiency' jetEta_4_variableBinning_numerator       jetEta_4_variableBinning_denominator",
        "effic_jetEta_5_variableBinning       'efficiency vs 5th jet eta; jet eta ; efficiency' jetEta_5_variableBinning_numerator       jetEta_5_variableBinning_denominator",
        "effic_jetEta_6_variableBinning       'efficiency vs 6th jet eta; jet eta ; efficiency' jetEta_6_variableBinning_numerator       jetEta_6_variableBinning_denominator",
        #
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetPt_2_variableBinning   'efficiency vs 2nd b-jet pt; bjet pt [GeV]; efficiency' bjetPt_2_variableBinning_numerator   bjetPt_2_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet eta; bjet eta ; efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet eta; bjet eta ; efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",
        #
        "effic_eventHT_variableBinning       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        #
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        #
        "effic_jetPtEta_1     'efficiency vs 1st jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetPtEta_2     'efficiency vs 2nd jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetPtEta_3     'efficiency vs 3rd jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_3_numerator       jetPtEta_3_denominator",
        "effic_jetPtEta_4     'efficiency vs 4th jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_4_numerator       jetPtEta_4_denominator",
        "effic_jetPtEta_5     'efficiency vs 5th jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_5_numerator       jetPtEta_5_denominator",
        "effic_jetPtEta_6     'efficiency vs 6th jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_6_numerator       jetPtEta_6_denominator",
        #
        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetEtaPhi_2    'efficiency vs 2nd jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_3    'efficiency vs 3rd jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_3_numerator       jetEtaPhi_3_denominator",
        "effic_jetEtaPhi_4    'efficiency vs 4th jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_4_numerator       jetEtaPhi_4_denominator",
        "effic_jetEtaPhi_5    'efficiency vs 5th jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_5_numerator       jetEtaPhi_5_denominator",
        "effic_jetEtaPhi_6    'efficiency vs 6th jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_6_numerator       jetEtaPhi_6_denominator",
        #
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",
        #
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_bjetEtaPhi_2    'efficiency vs 2nd b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",
        ),
)







topClient = cms.Sequence(
    topEfficiency_elejets
    + topEfficiency_eleHT
    + topEfficiency_singleMu
    + topEfficiency_diElec
    + topEfficiency_diMu
    + topEfficiency_ElecMu
    + fullyhadronicEfficiency_Reference
    + fullyhadronicEfficiency_DoubleBTag
    + fullyhadronicEfficiency_SingleBTag
)
