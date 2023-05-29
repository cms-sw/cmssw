import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

topEfficiency_elejets = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/EleJet/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met       'efficiency vs MET;MET [GeV];efficiency' met_numerator       met_denominator",
        "effic_elePt_1       'efficiency vs electron p_{T};electron p_{T} [GeV];efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron #eta;electron #eta;efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron #phi;electron #phi;efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_jetPt_1       'efficiency vs leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet #eta;jet #eta;efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet #phi;jet #phi;efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs electron p_{T};electron p_{T} [GeV];efficiency' elePt_1_variableBinning_numerator       elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs electron #eta;electron #eta;efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_jetPt_1_variableBinning       'efficiency vs leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetEta_1_variableBinning       'efficiency vs leading jet #eta;jet #eta;efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_eventHT_variableBinning       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_eleMulti       'efficiency vs electron multiplicity;electron multiplicity;efficiency' eleMulti_numerator       eleMulti_denominator",
        "effic_muMulti       'efficiency vs muon multiplicity;muon multiplicity;efficiency' muMulti_numerator       muMulti_denominator",
        "effic_elePtEta_1       'efficiency vs electron p_{T}-#eta;electron p_{T} [GeV];electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta-#phi;electron #eta;electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
        "effic_jetPtEta_1       'efficiency vs jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_elePt_jetPt       'efficiency vs electron p_{T} - jet p_{T};electron p_{T} [GeV];jet p_{T} [GeV]' elePt_jetPt_numerator       elePt_jetPt_denominator",
        "effic_elePt_eventHT       'efficiency vs electron p_{T} - event H_{T};electron p_{T} [GeV];event H_{T} [GeV]' elePt_eventHT_numerator       elePt_eventHT_denominator",

    ),
)

topEfficiency_eleHT = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/EleHT/*"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met       'efficiency vs MET;MET [GeV];efficiency' met_numerator       met_denominator",
        "effic_elePt_1       'efficiency vs electron p_{T};electron p_{T} [GeV];efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron #eta;electron #eta;efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron #phi;electron #phi;efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_jetPt_1       'efficiency vs leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet #eta;jet #eta;efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet #phi;jet #phi;efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_jetPt_2       'efficiency vs sub-leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'efficiency vs sub-leading jet #eta;jet #eta;efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'efficiency vs sub-leading jet #phi;jet #phi;efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi for HEP17;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs electron p_{T};electron p_{T} [GeV];efficiency' elePt_1_variableBinning_numerator       elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs electron #eta;electron #eta;efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_jetPt_1_variableBinning       'efficiency vs leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetEta_1_variableBinning       'efficiency vs leading jet #eta;jet #eta;efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs sub-leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs sub-leading jet #eta;jet #eta;efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_eventHT_variableBinning       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_eleMulti       'efficiency vs electron multiplicity;electron multiplicity;efficiency' eleMulti_numerator       eleMulti_denominator",
        "effic_muMulti       'efficiency vs muon multiplicity;muon multiplicity;efficiency' muMulti_numerator       muMulti_denominator",
        "effic_elePtEta_1       'efficiency vs electron p_{T}-#eta;electron p_{T} [GeV];electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta-#phi;electron #eta;electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
        "effic_jetPtEta_1       'efficiency vs jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetPtEta_2       'efficiency vs jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetEtaPhi_2       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_elePt_jetPt       'efficiency vs electron p_{T} - jet p_{T};electron p_{T} [GeV];jet p_{T} [GeV]' elePt_jetPt_numerator       elePt_jetPt_denominator",
        "effic_elePt_eventHT       'efficiency vs electron p_{T} - event H_{T};electron p_{T} [GeV];event H_{T} [GeV]' elePt_eventHT_numerator       elePt_eventHT_denominator",
    ),
)

topEfficiency_singleMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/SingleLepton/SingleMuon/*"),
    verbose        = cms.untracked.uint32(0),                                                                                                        
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met       'efficiency vs MET;MET [GeV];efficiency' met_numerator       met_denominator",
        "effic_metPhi       'efficiency vs MET #phi;MET #phi;efficiency' metPhi_numerator       metPhi_denominator",
        "effic_muPt_1       'efficiency vs muon p_{T};muon p_{T} [GeV];efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon #eta;muon #eta;efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon #phi;muon #phi;efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muEtaPhi_1     'efficiency vs muon #eta;muon #phi;efficiency' muEtaPhi_1_numerator       muEtaPhi_1_denominator",
        "effic_muPtEta_1     'efficiency vs muon p_{T};muon #eta;efficiency' muPtEta_1_numerator       muPtEta_1_denominator",
        "effic_jetPt_1       'efficiency vs leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet #eta;jet #eta;efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet #phi;jet #phi;efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_jetPt_2       'efficiency vs sub-leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'efficiency vs sub-leading jet #eta;jet #eta;efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'efficiency vs sub-leading jet #phi;jet #phi;efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetPtEta_1       'efficiency vs jet p_{T}-#eta;jet #eta;jet p_{T}' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetPtEta_2       'efficiency vs jet p_{T}-#eta;jet #eta;jet p_{T}' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetEtaPhi_2       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity;bjet multiplicity;efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_muMulti       'efficiency vs muon multiplicity;muon multiplicity;efficiency' muMulti_numerator       muMulti_denominator",
    ),
)

topEfficiency_diElec = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/DiLepton/DiElectron/*"),
    verbose        = cms.untracked.uint32(0),                                      
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met       'efficiency vs MET;MET [GeV];efficiency' met_numerator       met_denominator",
        "effic_metPhi       'efficiency vs MET #phi;MET #phi;efficiency' metPhi_numerator       metPhi_denominator",
        "effic_elePt_1       'efficiency vs electron p_{T};electron p_{T} [GeV];efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron #eta;electron #eta;efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron #phi;electron #phi;efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_2       'efficiency vs electron p_{T};electron p_{T} [GeV];efficiency' elePt_2_numerator       elePt_2_denominator",
        "effic_eleEta_2       'efficiency vs electron #eta;electron #eta;efficiency' eleEta_2_numerator       eleEta_2_denominator",
        "effic_elePhi_2       'efficiency vs electron #phi;electron #phi;efficiency' elePhi_2_numerator       elePhi_2_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta #phi;electron #eta #phi;efficiency' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
        "effic_elePtEta_1       'efficiency vs electron p_{T} #eta;electron p_{T} #eta;efficiency' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_2       'efficiency vs electron #eta #phi;electron #eta #phi;efficiency' eleEtaPhi_2_numerator       eleEtaPhi_2_denominator",
        "effic_elePtEta_2       'efficiency vs electron p_{T} #eta;electron p_{T} #eta;efficiency' elePtEta_2_numerator       elePtEta_2_denominator",
        "effic_jetPt_1       'efficiency vs leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet #eta;jet #eta;efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet #phi;jet #phi;efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_jetPt_2       'efficiency vs sub-leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'efficiency vs sub-leading jet #eta;jet #eta;efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'efficiency vs sub-leading jet #phi;jet #phi;efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetPtEta_1       'efficiency vs jet p_{T}-#eta;jet #eta;jet p_{T}' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetPtEta_2       'efficiency vs jet p_{T}-#eta;jet #eta;jet p_{T}' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetEtaPhi_2       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_ele1Pt_ele2Pt    'efficiency vs ele1-ele2 p_{T};ele1 p_{T};ele2 p_{T}' ele1Pt_ele2Pt_numerator       ele1Pt_ele2Pt_denominator",
        "effic_ele1Eta_ele2Eta    'efficiency vs ele1-ele2 #eta;ele1 #eta;ele2 #eta' ele1Eta_ele2Eta_numerator       ele1Eta_ele2Eta_denominator",
        "effic_ele1Phi_ele2Phi    'efficiency vs ele1-ele2 #phi;ele1 #phi;ele2 #phi' ele1Phi_ele2Phi_numerator       ele1Phi_ele2Phi_denominator",
        "effic_elePt_eventHT    'efficiency vs elePT-eventHT;ele p_{T};event H_{T}' elePt_eventHT_numerator       elePt_eventHT_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity;bjet multiplicity;efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_eleMulti       'efficiency vs ele multiplicity;ele multiplicity;efficiency' eleMulti_numerator       eleMulti_denominator",
    ),
)

topEfficiency_diMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/DiLepton/DiMuon/*"),
    verbose        = cms.untracked.uint32(0),                                      
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met       'efficiency vs MET;MET [GeV];efficiency' met_numerator       met_denominator",
        "effic_metPhi       'efficiency vs MET #phi;MET #phi;efficiency' metPhi_numerator       metPhi_denominator",
        "effic_muPt_1       'efficiency vs muon p_{T};muon p_{T} [GeV];efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon #eta;muon #eta;efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon #phi;muon #phi;efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_2       'efficiency vs muon p_{T};muon p_{T} [GeV];efficiency' muPt_2_numerator       muPt_2_denominator",
        "effic_muEta_2       'efficiency vs muon #eta;muon #eta;efficiency' muEta_2_numerator       muEta_2_denominator",
        "effic_muPhi_2       'efficiency vs muon #phi;muon #phi;efficiency' muPhi_2_numerator       muPhi_2_denominator",
        "effic_muEtaPhi_1       'efficiency vs muon #eta #phi;muon #eta #phi;efficiency' muEtaPhi_1_numerator       muEtaPhi_1_denominator",
        "effic_muPtEta_1       'efficiency vs muon p_{T} #eta;muon p_{T} #eta;efficiency' muPtEta_1_numerator       muPtEta_1_denominator",
        "effic_muEtaPhi_2       'efficiency vs muon #eta #phi;muon #eta #phi;efficiency' muEtaPhi_2_numerator       muEtaPhi_2_denominator",
        "effic_muPtEta_2       'efficiency vs muon p_{T} #eta;muon p_{T} #eta;efficiency' muPtEta_2_numerator       muPtEta_2_denominator",
        "effic_jetPt_1       'efficiency vs leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet #eta;jet #eta;efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet #phi;jet #phi;efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_jetPt_2       'efficiency vs sub-leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'efficiency vs sub-leading jet #eta;jet #eta;efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'efficiency vs sub-leading jet #phi;jet #phi;efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetPtEta_1       'efficiency vs jet p_{T}-#eta;jet #eta;jet p_{T}' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetPtEta_2       'efficiency vs jet p_{T}-#eta;jet #eta;jet p_{T}' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetEtaPhi_2       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_mu1Pt_mu2Pt    'efficiency vs mu1-mu2 p_{T};mu1 p_{T};mu2 p_{T}' mu1Pt_mu2Pt_numerator       mu1Pt_mu2Pt_denominator",
        "effic_mu1Eta_mu2Eta    'efficiency vs mu1-mu2 #eta;mu1 #eta;mu2 #eta' mu1Eta_mu2Eta_numerator       mu1Eta_mu2Eta_denominator",
        "effic_mu1Phi_mu2Phi    'efficiency vs mu1-mu2 #phi;mu1 #phi;mu2 #phi' mu1Phi_mu2Phi_numerator       mu1Phi_mu2Phi_denominator",
        "effic_muPt_eventHT    'efficiency vs muPT-eventHT;mu p_{T};event H_{T}' muPt_eventHT_numerator       muPt_eventHT_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity;bjet multiplicity;efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_muMulti       'efficiency vs muon multiplicity;muon multiplicity;efficiency' muMulti_numerator       muMulti_denominator",
    ),
)

topEfficiency_ElecMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/DiLepton/ElecMuon/*"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met       'efficiency vs MET;MET [GeV];efficiency' met_numerator       met_denominator",
        "effic_metPhi       'efficiency vs MET #phi;MET #phi;efficiency' metPhi_numerator       metPhi_denominator",
        "effic_muPt_1       'efficiency vs mu p_{T};mu p_{T} [GeV];efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs mu #eta;mu #eta;efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs mu #phi;mu #phi;efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_elePt_1       'efficiency vs electron p_{T};electron p_{T} [GeV];efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron #eta;electron #eta;efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron #phi;electron #phi;efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_muEtaPhi_1       'efficiency vs muon #eta #phi;muon #eta #phi;efficiency' muEtaPhi_1_numerator       muEtaPhi_1_denominator",
        "effic_muPtEta_1       'efficiency vs muon p_{T} #eta;muon p_{T} #eta;efficiency' muPtEta_1_numerator       muPtEta_1_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta #phi;electron #eta #phi;efficiency' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
        "effic_elePtEta_1       'efficiency vs electron p_{T} #eta;electron p_{T} #eta;efficiency' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_jetPt_1       'efficiency vs leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetEta_1       'efficiency vs leading jet #eta;jet #eta;efficiency' jetEta_1_numerator       jetEta_1_denominator",
        "effic_jetPhi_1       'efficiency vs leading jet #phi;jet #phi;efficiency' jetPhi_1_numerator       jetPhi_1_denominator",
        "effic_jetPt_2       'efficiency vs sub-leading jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetEta_2       'efficiency vs sub-leading jet #eta;jet #eta;efficiency' jetEta_2_numerator       jetEta_2_denominator",
        "effic_jetPhi_2       'efficiency vs sub-leading jet #phi;jet #phi;efficiency' jetPhi_2_numerator       jetPhi_2_denominator",
        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetPtEta_1       'efficiency vs jet p_{T}-#eta;jet #eta;jet p_{T}' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetEtaPhi_1       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetPtEta_2       'efficiency vs jet p_{T}-#eta;jet #eta;jet p_{T}' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetEtaPhi_2       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        "effic_elePt_muPt    'efficiency vs ele-mu p_{T};ele p_{T};mu p_{T}' elePt_muPt_numerator       elePt_muPt_denominator",
        "effic_eleEta_muEta    'efficiency vs ele-mu #eta;ele #eta;mu #phi' eleEta_muEta_numerator      eleEta_muEta_denominator",
        "effic_elePhi_muPhi    'efficiency vs ele-mu #phi;ele #phi;mu #phi' mu1Phi_mu2Phi_numerator       mu1Phi_mu2Phi_denominator",
        "effic_elePt_eventHT    'efficiency vs elePT-eventHT;ele p_{T};event H_{T}' elePt_eventHT_numerator       elePt_eventHT_denominator",
        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity;bjet multiplicity;efficiency' bjetMulti_numerator   bjetMulti_denominator",
        "effic_muMulti       'efficiency vs muon multiplicity;muon multiplicity;efficiency' muMulti_numerator       muMulti_denominator",
        "effic_eleMulti       'efficiency vs ele multiplicity;ele multiplicity;efficiency' eleMulti_numerator       eleMulti_denominator",
    ),
)

topEfficiency_fullyhadronic_Reference = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/FullyHadronic/Reference/*"),
    verbose        = cms.untracked.uint32(0), 
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_eventHT_variableBinning       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_variableBinning_numerator    eventHT_variableBinning_denominator",
    ),
)

topEfficiency_fullyhadronic_DoubleBTag = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/FullyHadronic/DoubleBTag/*"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt_1       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetPt_2       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetPt_3       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_numerator       jetPt_3_denominator",
        "effic_jetPt_4       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_numerator       jetPt_4_denominator",
        "effic_jetPt_5       'efficiency vs 5th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_5_numerator       jetPt_5_denominator",
        "effic_jetPt_6       'efficiency vs 6th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_6_numerator       jetPt_6_denominator",

        "effic_jetEta_1      'efficiency vs 1st jet #eta;jet #eta;efficiency' jetEta_1_numerator     jetEta_1_denominator",
        "effic_jetEta_2      'efficiency vs 2nd jet #eta;jet #eta;efficiency' jetEta_2_numerator     jetEta_2_denominator",
        "effic_jetEta_3      'efficiency vs 3rd jet #eta;jet #eta;efficiency' jetEta_3_numerator     jetEta_3_denominator",
        "effic_jetEta_4      'efficiency vs 4th jet #eta;jet #eta;efficiency' jetEta_4_numerator     jetEta_4_denominator",
        "effic_jetEta_5      'efficiency vs 5th jet #eta;jet #eta;efficiency' jetEta_5_numerator     jetEta_5_denominator",
        "effic_jetEta_6      'efficiency vs 6th jet #eta;jet #eta;efficiency' jetEta_6_numerator     jetEta_6_denominator",

        "effic_jetPhi_1      'efficiency vs 1st jet #phi;jet #phi;efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        "effic_jetPhi_2      'efficiency vs 2nd jet #phi;jet #phi;efficiency'    jetPhi_2_numerator      jetPhi_2_denominator",
        "effic_jetPhi_3      'efficiency vs 3rd jet #phi;jet #phi;efficiency'    jetPhi_3_numerator      jetPhi_3_denominator",
        "effic_jetPhi_4      'efficiency vs 4th jet #phi;jet #phi;efficiency'    jetPhi_4_numerator      jetPhi_4_denominator",
        "effic_jetPhi_5      'efficiency vs 5th jet #phi;jet #phi;efficiency'    jetPhi_5_numerator      jetPhi_5_denominator",
        "effic_jetPhi_6      'efficiency vs 6th jet #phi;jet #phi;efficiency'    jetPhi_6_numerator      jetPhi_6_denominator",

        "effic_bjetPt_1      'efficiency vs 1st b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetPt_2      'efficiency vs 2nd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_2_numerator  bjetPt_2_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet #eta;bjet #eta;efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetEta_2     'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency'  bjetEta_2_numerator   bjetEta_2_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet #phi;bjet #phi;efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet #phi;bjet #phi;efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet Discrim;bjet Discrim;efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet Discrim;bjet Discrim;efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",

        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",

        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetPt_3_variableBinning       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_variableBinning_numerator       jetPt_3_variableBinning_denominator",
        "effic_jetPt_4_variableBinning       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_variableBinning_numerator       jetPt_4_variableBinning_denominator",
        "effic_jetPt_5_variableBinning       'efficiency vs 5th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_5_variableBinning_numerator       jetPt_5_variableBinning_denominator",
        "effic_jetPt_6_variableBinning       'efficiency vs 6th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_6_variableBinning_numerator       jetPt_6_variableBinning_denominator",

        "effic_jetEta_1_variableBinning       'efficiency vs 1st jet #eta;jet #eta;efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs 2nd jet #eta;jet #eta;efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_jetEta_3_variableBinning       'efficiency vs 3rd jet #eta;jet #eta;efficiency' jetEta_3_variableBinning_numerator       jetEta_3_variableBinning_denominator",
        "effic_jetEta_4_variableBinning       'efficiency vs 4th jet #eta;jet #eta;efficiency' jetEta_4_variableBinning_numerator       jetEta_4_variableBinning_denominator",
        "effic_jetEta_5_variableBinning       'efficiency vs 5th jet #eta;jet #eta;efficiency' jetEta_5_variableBinning_numerator       jetEta_5_variableBinning_denominator",
        "effic_jetEta_6_variableBinning       'efficiency vs 6th jet #eta;jet #eta;efficiency' jetEta_6_variableBinning_numerator       jetEta_6_variableBinning_denominator",

        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetPt_2_variableBinning   'efficiency vs 2nd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_2_variableBinning_numerator   bjetPt_2_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet #eta;bjet #eta;efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",

        "effic_eventHT_variableBinning       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",

        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity;bjet multiplicity;efficiency' bjetMulti_numerator   bjetMulti_denominator",

        "effic_jetPtEta_1     'efficiency vs 1st jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetPtEta_2     'efficiency vs 2nd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetPtEta_3     'efficiency vs 3rd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_3_numerator       jetPtEta_3_denominator",
        "effic_jetPtEta_4     'efficiency vs 4th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_4_numerator       jetPtEta_4_denominator",
        "effic_jetPtEta_5     'efficiency vs 5th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_5_numerator       jetPtEta_5_denominator",
        "effic_jetPtEta_6     'efficiency vs 6th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_6_numerator       jetPtEta_6_denominator",

        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetEtaPhi_2    'efficiency vs 2nd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_3    'efficiency vs 3rd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_3_numerator       jetEtaPhi_3_denominator",
        "effic_jetEtaPhi_4    'efficiency vs 4th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_4_numerator       jetEtaPhi_4_denominator",
        "effic_jetEtaPhi_5    'efficiency vs 5th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_5_numerator       jetEtaPhi_5_denominator",
        "effic_jetEtaPhi_6    'efficiency vs 6th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_6_numerator       jetEtaPhi_6_denominator",

        "effic_bjetPtEta_1    'efficiency vs 1st b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",

        "effic_bjetEtaPhi_1   'efficiency vs 1st b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_bjetEtaPhi_2   'efficiency vs 2nd b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",

        "effic_bjetCSVHT_1 'efficiency vs 1st b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_1_numerator bjetCSVHT_1_denominator"
        "effic_bjetCSVHT_2 'efficiency vs 2nd b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_2_numerator bjetCSVHT_2_denominator"
    ),
)

topEfficiency_fullyhadronic_DoubleBTag_DeepJet = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/FullyHadronic/DoubleBTagDeepJet/*"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt_1       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetPt_2       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetPt_3       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_numerator       jetPt_3_denominator",
        "effic_jetPt_4       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_numerator       jetPt_4_denominator",
        "effic_jetPt_5       'efficiency vs 5th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_5_numerator       jetPt_5_denominator",
        "effic_jetPt_6       'efficiency vs 6th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_6_numerator       jetPt_6_denominator",

        "effic_jetEta_1      'efficiency vs 1st jet #eta;jet #eta;efficiency' jetEta_1_numerator     jetEta_1_denominator",
        "effic_jetEta_2      'efficiency vs 2nd jet #eta;jet #eta;efficiency' jetEta_2_numerator     jetEta_2_denominator",
        "effic_jetEta_3      'efficiency vs 3rd jet #eta;jet #eta;efficiency' jetEta_3_numerator     jetEta_3_denominator",
        "effic_jetEta_4      'efficiency vs 4th jet #eta;jet #eta;efficiency' jetEta_4_numerator     jetEta_4_denominator",
        "effic_jetEta_5      'efficiency vs 5th jet #eta;jet #eta;efficiency' jetEta_5_numerator     jetEta_5_denominator",
        "effic_jetEta_6      'efficiency vs 6th jet #eta;jet #eta;efficiency' jetEta_6_numerator     jetEta_6_denominator",

        "effic_jetPhi_1      'efficiency vs 1st jet #phi;jet #phi;efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        "effic_jetPhi_2      'efficiency vs 2nd jet #phi;jet #phi;efficiency'    jetPhi_2_numerator      jetPhi_2_denominator",
        "effic_jetPhi_3      'efficiency vs 3rd jet #phi;jet #phi;efficiency'    jetPhi_3_numerator      jetPhi_3_denominator",
        "effic_jetPhi_4      'efficiency vs 4th jet #phi;jet #phi;efficiency'    jetPhi_4_numerator      jetPhi_4_denominator",
        "effic_jetPhi_5      'efficiency vs 5th jet #phi;jet #phi;efficiency'    jetPhi_5_numerator      jetPhi_5_denominator",
        "effic_jetPhi_6      'efficiency vs 6th jet #phi;jet #phi;efficiency'    jetPhi_6_numerator      jetPhi_6_denominator",

        "effic_bjetPt_1      'efficiency vs 1st b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetPt_2      'efficiency vs 2nd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_2_numerator  bjetPt_2_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet #eta;bjet #eta;efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetEta_2     'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency'  bjetEta_2_numerator   bjetEta_2_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet #phi;bjet #phi;efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet #phi;bjet #phi;efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet Discrim;bjet Discrim;efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet Discrim;bjet Discrim;efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",

        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",

        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetPt_3_variableBinning       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_variableBinning_numerator       jetPt_3_variableBinning_denominator",
        "effic_jetPt_4_variableBinning       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_variableBinning_numerator       jetPt_4_variableBinning_denominator",
        "effic_jetPt_5_variableBinning       'efficiency vs 5th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_5_variableBinning_numerator       jetPt_5_variableBinning_denominator",
        "effic_jetPt_6_variableBinning       'efficiency vs 6th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_6_variableBinning_numerator       jetPt_6_variableBinning_denominator",

        "effic_jetEta_1_variableBinning       'efficiency vs 1st jet #eta;jet #eta;efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs 2nd jet #eta;jet #eta;efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_jetEta_3_variableBinning       'efficiency vs 3rd jet #eta;jet #eta;efficiency' jetEta_3_variableBinning_numerator       jetEta_3_variableBinning_denominator",
        "effic_jetEta_4_variableBinning       'efficiency vs 4th jet #eta;jet #eta;efficiency' jetEta_4_variableBinning_numerator       jetEta_4_variableBinning_denominator",
        "effic_jetEta_5_variableBinning       'efficiency vs 5th jet #eta;jet #eta;efficiency' jetEta_5_variableBinning_numerator       jetEta_5_variableBinning_denominator",
        "effic_jetEta_6_variableBinning       'efficiency vs 6th jet #eta;jet #eta;efficiency' jetEta_6_variableBinning_numerator       jetEta_6_variableBinning_denominator",

        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetPt_2_variableBinning   'efficiency vs 2nd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_2_variableBinning_numerator   bjetPt_2_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet #eta;bjet #eta;efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",

        "effic_eventHT_variableBinning       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",

        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity;bjet multiplicity;efficiency' bjetMulti_numerator   bjetMulti_denominator",

        "effic_jetPtEta_1     'efficiency vs 1st jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetPtEta_2     'efficiency vs 2nd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetPtEta_3     'efficiency vs 3rd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_3_numerator       jetPtEta_3_denominator",
        "effic_jetPtEta_4     'efficiency vs 4th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_4_numerator       jetPtEta_4_denominator",
        "effic_jetPtEta_5     'efficiency vs 5th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_5_numerator       jetPtEta_5_denominator",
        "effic_jetPtEta_6     'efficiency vs 6th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_6_numerator       jetPtEta_6_denominator",

        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetEtaPhi_2    'efficiency vs 2nd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_3    'efficiency vs 3rd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_3_numerator       jetEtaPhi_3_denominator",
        "effic_jetEtaPhi_4    'efficiency vs 4th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_4_numerator       jetEtaPhi_4_denominator",
        "effic_jetEtaPhi_5    'efficiency vs 5th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_5_numerator       jetEtaPhi_5_denominator",
        "effic_jetEtaPhi_6    'efficiency vs 6th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_6_numerator       jetEtaPhi_6_denominator",

        "effic_bjetPtEta_1    'efficiency vs 1st b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",

        "effic_bjetEtaPhi_1   'efficiency vs 1st b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_bjetEtaPhi_2   'efficiency vs 2nd b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",

        "effic_bjetCSVHT_1 'efficiency vs 1st b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_1_numerator bjetCSVHT_1_denominator"
        "effic_bjetCSVHT_2 'efficiency vs 2nd b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_2_numerator bjetCSVHT_2_denominator"
    ),
)

topEfficiency_fullyhadronic_SingleBTag = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/FullyHadronic/SingleBTag/*"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt_1       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetPt_2       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetPt_3       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_numerator       jetPt_3_denominator",
        "effic_jetPt_4       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_numerator       jetPt_4_denominator",
        "effic_jetPt_5       'efficiency vs 5th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_5_numerator       jetPt_5_denominator",
        "effic_jetPt_6       'efficiency vs 6th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_6_numerator       jetPt_6_denominator",

        "effic_jetEta_1      'efficiency vs 1st jet #eta;jet #eta;efficiency' jetEta_1_numerator     jetEta_1_denominator",
        "effic_jetEta_2      'efficiency vs 2nd jet #eta;jet #eta;efficiency' jetEta_2_numerator     jetEta_2_denominator",
        "effic_jetEta_3      'efficiency vs 3rd jet #eta;jet #eta;efficiency' jetEta_3_numerator     jetEta_3_denominator",
        "effic_jetEta_4      'efficiency vs 4th jet #eta;jet #eta;efficiency' jetEta_4_numerator     jetEta_4_denominator",
        "effic_jetEta_5      'efficiency vs 5th jet #eta;jet #eta;efficiency' jetEta_5_numerator     jetEta_5_denominator",
        "effic_jetEta_6      'efficiency vs 6th jet #eta;jet #eta;efficiency' jetEta_6_numerator     jetEta_6_denominator",

        "effic_jetPhi_1      'efficiency vs 1st jet #phi;jet #phi;efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        "effic_jetPhi_2      'efficiency vs 2nd jet #phi;jet #phi;efficiency'    jetPhi_2_numerator      jetPhi_2_denominator",
        "effic_jetPhi_3      'efficiency vs 3rd jet #phi;jet #phi;efficiency'    jetPhi_3_numerator      jetPhi_3_denominator",
        "effic_jetPhi_4      'efficiency vs 4th jet #phi;jet #phi;efficiency'    jetPhi_4_numerator      jetPhi_4_denominator",
        "effic_jetPhi_5      'efficiency vs 5th jet #phi;jet #phi;efficiency'    jetPhi_5_numerator      jetPhi_5_denominator",
        "effic_jetPhi_6      'efficiency vs 6th jet #phi;jet #phi;efficiency'    jetPhi_6_numerator      jetPhi_6_denominator",

        "effic_bjetPt_1      'efficiency vs 1st b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetPt_2      'efficiency vs 2nd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_2_numerator  bjetPt_2_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet #eta;bjet #eta;efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetEta_2     'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency'  bjetEta_2_numerator   bjetEta_2_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet #phi;bjet #phi;efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet #phi;bjet #phi;efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet Discrim;bjet Discrim;efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet Discrim;bjet Discrim;efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",

        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",

        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetPt_3_variableBinning       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_variableBinning_numerator       jetPt_3_variableBinning_denominator",
        "effic_jetPt_4_variableBinning       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_variableBinning_numerator       jetPt_4_variableBinning_denominator",
        "effic_jetPt_5_variableBinning       'efficiency vs 5th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_5_variableBinning_numerator       jetPt_5_variableBinning_denominator",
        "effic_jetPt_6_variableBinning       'efficiency vs 6th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_6_variableBinning_numerator       jetPt_6_variableBinning_denominator",

        "effic_jetEta_1_variableBinning       'efficiency vs 1st jet #eta;jet #eta;efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs 2nd jet #eta;jet #eta;efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_jetEta_3_variableBinning       'efficiency vs 3rd jet #eta;jet #eta;efficiency' jetEta_3_variableBinning_numerator       jetEta_3_variableBinning_denominator",
        "effic_jetEta_4_variableBinning       'efficiency vs 4th jet #eta;jet #eta;efficiency' jetEta_4_variableBinning_numerator       jetEta_4_variableBinning_denominator",
        "effic_jetEta_5_variableBinning       'efficiency vs 5th jet #eta;jet #eta;efficiency' jetEta_5_variableBinning_numerator       jetEta_5_variableBinning_denominator",
        "effic_jetEta_6_variableBinning       'efficiency vs 6th jet #eta;jet #eta;efficiency' jetEta_6_variableBinning_numerator       jetEta_6_variableBinning_denominator",

        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetPt_2_variableBinning   'efficiency vs 2nd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_2_variableBinning_numerator   bjetPt_2_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet #eta;bjet #eta;efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",

        "effic_eventHT_variableBinning    'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",

        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity;bjet multiplicity;efficiency' bjetMulti_numerator   bjetMulti_denominator",

        "effic_jetPtEta_1     'efficiency vs 1st jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetPtEta_2     'efficiency vs 2nd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetPtEta_3     'efficiency vs 3rd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_3_numerator       jetPtEta_3_denominator",
        "effic_jetPtEta_4     'efficiency vs 4th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_4_numerator       jetPtEta_4_denominator",
        "effic_jetPtEta_5     'efficiency vs 5th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_5_numerator       jetPtEta_5_denominator",
        "effic_jetPtEta_6     'efficiency vs 6th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_6_numerator       jetPtEta_6_denominator",

        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetEtaPhi_2    'efficiency vs 2nd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_3    'efficiency vs 3rd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_3_numerator       jetEtaPhi_3_denominator",
        "effic_jetEtaPhi_4    'efficiency vs 4th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_4_numerator       jetEtaPhi_4_denominator",
        "effic_jetEtaPhi_5    'efficiency vs 5th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_5_numerator       jetEtaPhi_5_denominator",
        "effic_jetEtaPhi_6    'efficiency vs 6th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_6_numerator       jetEtaPhi_6_denominator",

        "effic_bjetPtEta_1    'efficiency vs 1st b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_bjetEtaPhi_2    'efficiency vs 2nd b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",

        "effic_bjetCSVHT_1 'efficiency vs 1st b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_1_numerator bjetCSVHT_1_denominator"
        "effic_bjetCSVHT_2 'efficiency vs 2nd b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_2_numerator bjetCSVHT_2_denominator"
    ),
)


topEfficiency_fullyhadronic_SingleBTag_DeepJet = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/FullyHadronic/SingleBTagDeepJet/*"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt_1       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetPt_2       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetPt_3       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_numerator       jetPt_3_denominator",
        "effic_jetPt_4       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_numerator       jetPt_4_denominator",
        "effic_jetPt_5       'efficiency vs 5th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_5_numerator       jetPt_5_denominator",
        "effic_jetPt_6       'efficiency vs 6th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_6_numerator       jetPt_6_denominator",

        "effic_jetEta_1      'efficiency vs 1st jet #eta;jet #eta;efficiency' jetEta_1_numerator     jetEta_1_denominator",
        "effic_jetEta_2      'efficiency vs 2nd jet #eta;jet #eta;efficiency' jetEta_2_numerator     jetEta_2_denominator",
        "effic_jetEta_3      'efficiency vs 3rd jet #eta;jet #eta;efficiency' jetEta_3_numerator     jetEta_3_denominator",
        "effic_jetEta_4      'efficiency vs 4th jet #eta;jet #eta;efficiency' jetEta_4_numerator     jetEta_4_denominator",
        "effic_jetEta_5      'efficiency vs 5th jet #eta;jet #eta;efficiency' jetEta_5_numerator     jetEta_5_denominator",
        "effic_jetEta_6      'efficiency vs 6th jet #eta;jet #eta;efficiency' jetEta_6_numerator     jetEta_6_denominator",

        "effic_jetPhi_1      'efficiency vs 1st jet #phi;jet #phi;efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        "effic_jetPhi_2      'efficiency vs 2nd jet #phi;jet #phi;efficiency'    jetPhi_2_numerator      jetPhi_2_denominator",
        "effic_jetPhi_3      'efficiency vs 3rd jet #phi;jet #phi;efficiency'    jetPhi_3_numerator      jetPhi_3_denominator",
        "effic_jetPhi_4      'efficiency vs 4th jet #phi;jet #phi;efficiency'    jetPhi_4_numerator      jetPhi_4_denominator",
        "effic_jetPhi_5      'efficiency vs 5th jet #phi;jet #phi;efficiency'    jetPhi_5_numerator      jetPhi_5_denominator",
        "effic_jetPhi_6      'efficiency vs 6th jet #phi;jet #phi;efficiency'    jetPhi_6_numerator      jetPhi_6_denominator",

        "effic_bjetPt_1      'efficiency vs 1st b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetPt_2      'efficiency vs 2nd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_2_numerator  bjetPt_2_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet #eta;bjet #eta;efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetEta_2     'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency'  bjetEta_2_numerator   bjetEta_2_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet #phi;bjet #phi;efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet #phi;bjet #phi;efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet Discrim;bjet Discrim;efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet Discrim;bjet Discrim;efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",

        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",

        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetPt_3_variableBinning       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_variableBinning_numerator       jetPt_3_variableBinning_denominator",
        "effic_jetPt_4_variableBinning       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_variableBinning_numerator       jetPt_4_variableBinning_denominator",
        "effic_jetPt_5_variableBinning       'efficiency vs 5th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_5_variableBinning_numerator       jetPt_5_variableBinning_denominator",
        "effic_jetPt_6_variableBinning       'efficiency vs 6th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_6_variableBinning_numerator       jetPt_6_variableBinning_denominator",

        "effic_jetEta_1_variableBinning       'efficiency vs 1st jet #eta;jet #eta;efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs 2nd jet #eta;jet #eta;efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_jetEta_3_variableBinning       'efficiency vs 3rd jet #eta;jet #eta;efficiency' jetEta_3_variableBinning_numerator       jetEta_3_variableBinning_denominator",
        "effic_jetEta_4_variableBinning       'efficiency vs 4th jet #eta;jet #eta;efficiency' jetEta_4_variableBinning_numerator       jetEta_4_variableBinning_denominator",
        "effic_jetEta_5_variableBinning       'efficiency vs 5th jet #eta;jet #eta;efficiency' jetEta_5_variableBinning_numerator       jetEta_5_variableBinning_denominator",
        "effic_jetEta_6_variableBinning       'efficiency vs 6th jet #eta;jet #eta;efficiency' jetEta_6_variableBinning_numerator       jetEta_6_variableBinning_denominator",

        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetPt_2_variableBinning   'efficiency vs 2nd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_2_variableBinning_numerator   bjetPt_2_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet #eta;bjet #eta;efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",

        "effic_eventHT_variableBinning    'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",

        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity;bjet multiplicity;efficiency' bjetMulti_numerator   bjetMulti_denominator",

        "effic_jetPtEta_1     'efficiency vs 1st jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetPtEta_2     'efficiency vs 2nd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetPtEta_3     'efficiency vs 3rd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_3_numerator       jetPtEta_3_denominator",
        "effic_jetPtEta_4     'efficiency vs 4th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_4_numerator       jetPtEta_4_denominator",
        "effic_jetPtEta_5     'efficiency vs 5th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_5_numerator       jetPtEta_5_denominator",
        "effic_jetPtEta_6     'efficiency vs 6th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_6_numerator       jetPtEta_6_denominator",

        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetEtaPhi_2    'efficiency vs 2nd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_3    'efficiency vs 3rd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_3_numerator       jetEtaPhi_3_denominator",
        "effic_jetEtaPhi_4    'efficiency vs 4th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_4_numerator       jetEtaPhi_4_denominator",
        "effic_jetEtaPhi_5    'efficiency vs 5th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_5_numerator       jetEtaPhi_5_denominator",
        "effic_jetEtaPhi_6    'efficiency vs 6th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_6_numerator       jetEtaPhi_6_denominator",

        "effic_bjetPtEta_1    'efficiency vs 1st b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_bjetEtaPhi_2    'efficiency vs 2nd b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",

        "effic_bjetCSVHT_1 'efficiency vs 1st b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_1_numerator bjetCSVHT_1_denominator"
        "effic_bjetCSVHT_2 'efficiency vs 2nd b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_2_numerator bjetCSVHT_2_denominator"
    ),
)

topEfficiency_fullyhadronic_TripleBTag = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/FullyHadronic/TripleBTag/*"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt_1       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetPt_2       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetPt_3       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_numerator       jetPt_3_denominator",
        "effic_jetPt_4       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_numerator       jetPt_4_denominator",

        "effic_jetEta_1      'efficiency vs 1st jet #eta;jet #eta;efficiency' jetEta_1_numerator     jetEta_1_denominator",
        "effic_jetEta_2      'efficiency vs 2nd jet #eta;jet #eta;efficiency' jetEta_2_numerator     jetEta_2_denominator",
        "effic_jetEta_3      'efficiency vs 3rd jet #eta;jet #eta;efficiency' jetEta_3_numerator     jetEta_3_denominator",
        "effic_jetEta_4      'efficiency vs 4th jet #eta;jet #eta;efficiency' jetEta_4_numerator     jetEta_4_denominator",

        "effic_jetPhi_1      'efficiency vs 1st jet #phi;jet #phi;efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        "effic_jetPhi_2      'efficiency vs 2nd jet #phi;jet #phi;efficiency'    jetPhi_2_numerator      jetPhi_2_denominator",
        "effic_jetPhi_3      'efficiency vs 3rd jet #phi;jet #phi;efficiency'    jetPhi_3_numerator      jetPhi_3_denominator",
        "effic_jetPhi_4      'efficiency vs 4th jet #phi;jet #phi;efficiency'    jetPhi_4_numerator      jetPhi_4_denominator",

        "effic_bjetPt_1      'efficiency vs 1st b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetPt_2      'efficiency vs 2nd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_2_numerator  bjetPt_2_denominator",
        "effic_bjetPt_3      'efficiency vs 3rd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_3_numerator  bjetPt_3_denominator",
        "effic_bjetPt_4      'efficiency vs 4th b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_4_numerator  bjetPt_4_denominator",

        "effic_bjetEta_1     'efficiency vs 1st b-jet #eta;bjet #eta;efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetEta_2     'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency'  bjetEta_2_numerator   bjetEta_2_denominator",
        "effic_bjetEta_3     'efficiency vs 3rd b-jet #eta;bjet #eta;efficiency'  bjetEta_3_numerator   bjetEta_3_denominator",
        "effic_bjetEta_4     'efficiency vs 4th b-jet #eta;bjet #eta;efficiency'  bjetEta_4_numerator   bjetEta_4_denominator",

        "effic_bjetPhi_1     'efficiency vs 1st b-jet #phi;bjet #phi;efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet #phi;bjet #phi;efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetPhi_3     'efficiency vs 3rd b-jet #phi;bjet #phi;efficiency'  bjetPhi_3_numerator   bjetPhi_3_denominator",
        "effic_bjetPhi_4     'efficiency vs 4th b-jet #phi;bjet #phi;efficiency'  bjetPhi_4_numerator   bjetPhi_4_denominator",

        "effic_bjetCSV_1     'efficiency vs 1st b-jet Discrim;bjet Discrim;efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet Discrim;bjet Discrim;efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",
        "effic_bjetCSV_3     'efficiency vs 3nd b-jet Discrim;bjet Discrim;efficiency' bjetCSV_3_numerator  bjetCSV_3_denominator",
        "effic_bjetCSV_4     'efficiency vs 4nd b-jet Discrim;bjet Discrim;efficiency' bjetCSV_4_numerator  bjetCSV_4_denominator",

        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",

        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetPt_3_variableBinning       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_variableBinning_numerator       jetPt_3_variableBinning_denominator",
        "effic_jetPt_4_variableBinning       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_variableBinning_numerator       jetPt_4_variableBinning_denominator",

        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet #eta;bjet #eta;efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",
        "effic_bjetEta_3_variableBinning  'efficiency vs 3rd b-jet #eta;bjet #eta;efficiency' bjetEta_3_variableBinning_numerator     bjetEta_3_variableBinning_denominator",
        "effic_bjetEta_4_variableBinning  'efficiency vs 4th b-jet #eta;bjet #eta;efficiency' bjetEta_4_variableBinning_numerator     bjetEta_4_variableBinning_denominator",

        "effic_eventHT_variableBinning    'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",

        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity;bjet multiplicity;efficiency' bjetMulti_numerator   bjetMulti_denominator",

        "effic_jetPtEta_1     'efficiency vs 1st jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetPtEta_2     'efficiency vs 2nd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetPtEta_3     'efficiency vs 3rd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_3_numerator       jetPtEta_3_denominator",
        "effic_jetPtEta_4     'efficiency vs 4th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_4_numerator       jetPtEta_4_denominator",

        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetEtaPhi_2    'efficiency vs 2nd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_3    'efficiency vs 3rd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_3_numerator       jetEtaPhi_3_denominator",
        "effic_jetEtaPhi_4    'efficiency vs 4th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_4_numerator       jetEtaPhi_4_denominator",

        "effic_bjetPtEta_1    'efficiency vs 1st b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",
        "effic_bjetPtEta_3    'efficiency vs 3rd b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_3_numerator   bjetPtEta_3_denominator",
        "effic_bjetPtEta_4    'efficiency vs 4th b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_4_numerator   bjetPtEta_4_denominator",

        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_bjetEtaPhi_2    'efficiency vs 2nd b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",
        "effic_bjetEtaPhi_3    'efficiency vs 3rd b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_3_numerator  bjetEtaPhi_3_denominator",
        "effic_bjetEtaPhi_4    'efficiency vs 4th b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_4_numerator  bjetEtaPhi_4_denominator",

        "effic_bjetCSVHT_1 'efficiency vs 1st b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_1_numerator bjetCSVHT_1_denominator"
        "effic_bjetCSVHT_2 'efficiency vs 2nd b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_2_numerator bjetCSVHT_2_denominator"
        "effic_bjetCSVHT_3 'efficiency vs 3rd b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_3_numerator bjetCSVHT_3_denominator"
        "effic_bjetCSVHT_4 'efficiency vs 4th b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_4_numerator bjetCSVHT_4_denominator"
    ),
)

topClient = cms.Sequence(
    topEfficiency_elejets
  + topEfficiency_eleHT
  + topEfficiency_singleMu
  + topEfficiency_diElec
  + topEfficiency_diMu
  + topEfficiency_ElecMu
  + topEfficiency_fullyhadronic_Reference
  + topEfficiency_fullyhadronic_DoubleBTag
  + topEfficiency_fullyhadronic_DoubleBTag_DeepJet
  + topEfficiency_fullyhadronic_SingleBTag
  + topEfficiency_fullyhadronic_SingleBTag_DeepJet
  + topEfficiency_fullyhadronic_TripleBTag
)
