import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester


topEfficiency_elejets = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/Top/EleJet/*"),
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
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/Top/EleHT/*"),
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


topEfficiency_diElec = DQMEDHarvester("DQMGenericClient",
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
        "effic_ele1Pt_ele2Pt    'efficiency vs ele1-ele2 pt; ele1 pt; ele2 pt' ele1Pt_ele2Pt_numerator       ele1Pt_ele2Pt_denominator",
        "effic_ele1Eta_ele2Eta    'efficiency vs ele1-ele2 #eta; ele1 #eta; ele2 #eta' ele1Eta_ele2Eta_numerator       ele1Eta_ele2Eta_denominator",


    ),
)



topEfficiency_diMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/Top/DiLepton/DiMuon/"),
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
    subDirs        = cms.untracked.vstring("HLT/TopHLTOffline/TopMonitor/Top/DiLepton/ElecMuon/"),
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


singleTopEfficiency_singleMu = DQMEDHarvester("DQMGenericClient",
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
