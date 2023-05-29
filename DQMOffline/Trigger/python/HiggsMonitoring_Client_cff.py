import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQMOffline.Trigger.VBFMETMonitor_Client_cff import *
from DQMOffline.Trigger.HMesonGammaMonitor_Client_cff import *
from DQMOffline.Trigger.VBFTauMonitor_Client_cff import *
from DQMOffline.Trigger.MssmHbbBtagTriggerMonitor_Client_cfi import *
from DQMOffline.Trigger.MssmHbbMonitoring_Client_cfi import *
from DQMOffline.Trigger.PhotonMonitor_cff import *

metbtagEfficiency_met = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/*"),
    subDirs        = cms.untracked.vstring("HLT/HIG/*"),
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

metbtagEfficiency_btag = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/*"),
    subDirs        = cms.untracked.vstring("HLT/HIG/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt_1       'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_numerator       jetPt_1_denominator",
        #
        "effic_jetEta_1      'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_numerator     jetEta_1_denominator",
        #
        "effic_jetPhi_1      'efficiency vs 1st jet phi; jet phi ; efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        #
        "effic_bjetPt_1      'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetEta_1     'efficiency vs 1st b-jet eta; bjet eta ; efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        #
        "effic_eventHT       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi; jet #eta; jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",
        #
        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        #
        "effic_jetEta_1_variableBinning       'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        #
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        #
        "effic_eventHT_variableBinning       'efficiency vs event HT; event HT [GeV]; efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",
        #
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        #
        "effic_jetPtEta_1     'efficiency vs 1st jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        #
        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        #
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        #
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        #
        "effic_bjetCSVHT_1 'efficiency vs 1st b-jet csv - event HT; bjet csv ; event HT [GeV]' bjetCSVHT_1_numerator bjetCSVHT_1_denominator"
        ),
)

###############Same flavour dilepton with dz cuts#######################
ele23Ele12CaloIdLTrackIdLIsoVL_effdz = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/DiLepton/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/DiLepton/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_elePt_1        'efficiency vs lead electron pt; electron pt [GeV]; efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs lead electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs lead electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs lead electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator       elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning      'efficiency vs lead electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_elePtEta_1       'efficiency vs lead electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1      'efficiency vs lead electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
        "effic_elePt_2       'efficiency vs sub-lead electron pt; electron pt [GeV]; efficiency' elePt_2_numerator       elePt_2_denominator",
        "effic_eleEta_2       'efficiency vs sub-lead electron eta; electron eta ; efficiency' eleEta_2_numerator       eleEta_2_denominator",
        "effic_elePhi_2       'efficiency vs sub-lead electron phi; electron phi ; efficiency' elePhi_2_numerator       elePhi_2_denominator",
        "effic_elePt_2_variableBinning       'efficiency vs sub-lead electron pt; electron pt [GeV]; efficiency' elePt_2_variableBinning_numerator       elePt_2_variableBinning_denominator",
        "effic_eleEta_2_variableBinning       'efficiency vs sub-lead electron eta; electron eta ; efficiency' eleEta_2_variableBinning_numerator       eleEta_2_variableBinning_denominator",
        "effic_elePtEta_2       'efficiency vs sub-lead electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_2_numerator       elePtEta_2_denominator",
        "effic_eleEtaPhi_2      'efficiency vs sub-lead electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_2_numerator       eleEtaPhi_2_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_ElectronPt_vs_LS 'Lead electron p_T efficiency vs LS; LS; Electron p_T efficiency' eleVsLS_numerator eleVsLS_denominator"
    ),
)

################################MuEG cross triggers###################################
muEleDz_effele =  DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/HIG/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg/",
                                           "HLT/HIG/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg/"
                                          ),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_elePt_1       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator       elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_elePtEta_1       'efficiency vs electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
    	"effic_ElectronPt_vs_LS 'Electron p_T efficiency vs LS; LS; Electron p_T efficiency' eleVsLS_numerator eleVsLS_denominator"
    ),
)
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_effele =  DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_elePt_1       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator       elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_elePtEta_1       'efficiency vs electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
    	"effic_ElectronPt_vs_LS 'Electron p_T efficiency vs LS; LS; Electron p_T efficiency' eleVsLS_numerator eleVsLS_denominator"
    ),
)
muEleDz_effmu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/HIG/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/muLeg/",
                                           "HLT/HIG/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/muLeg/"
                                          ),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPtEta_1       'efficiency vs muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator       muPtEta_1_denominator",
        "effic_muEtaPhi_1       'efficiency vs muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator       muEtaPhi_1_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_MuonPt_vs_LS 'Muon p_T efficiency vs LS; LS; Muon p_T efficiency' muVsLS_numerator muVsLS_denominator"
    ),
)
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_effmu = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/muLeg/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/muLeg/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPtEta_1       'efficiency vs muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator       muPtEta_1_denominator",
        "effic_muEtaPhi_1       'efficiency vs muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator       muEtaPhi_1_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_MuonPt_vs_LS 'Muon p_T efficiency vs LS; LS; Muon p_T efficiency' muVsLS_numerator muVsLS_denominator"
    ),
)

mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_effele =  DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_elePt_1       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator       elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_elePtEta_1       'efficiency vs electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
    	"effic_ElectronPt_vs_LS 'Electron p_T efficiency vs LS; LS; Electron p_T efficiency' eleVsLS_numerator eleVsLS_denominator"
    ),
)

mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_effmu = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/muLeg/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/muLeg/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPtEta_1       'efficiency vs muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator       muPtEta_1_denominator",
        "effic_muEtaPhi_1       'efficiency vs muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator       muEtaPhi_1_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_MuonPt_vs_LS 'Muon p_T efficiency vs LS; LS; Muon p_T efficiency' muVsLS_numerator muVsLS_denominator"
    ),
)

##########################Triple Electron################################3##
ele16ele12ele8caloIdLTrackIdL =  DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/TriLepton/HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/TriLepton/HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_elePt_1       'efficiency vs lead electron pt; electron pt [GeV]; efficiency' elePt_1_numerator	elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs lead electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs lead electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs lead electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator	elePt_1variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs lead electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_elePtEta_1	'efficiency vs lead electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1	 'efficiency vs lead electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
        "effic_elePt_2       'efficiency vs sub-leading electron pt; electron pt [GeV]; efficiency' elePt_2_numerator	elePt_2_denominator",
        "effic_eleEta_2       'efficiency vs sub-leading electron eta; electron eta ; efficiency' eleEta_2_numerator       eleEta_2_denominator",
        "effic_elePhi_2       'efficiency vs sub-leading electron phi; electron phi ; efficiency' elePhi_2_numerator       elePhi_2_denominator",
        "effic_elePt_2_variableBinning       'efficiency vs sub-leading electron pt; electron pt [GeV]; efficiency' elePt_2_variableBinning_numerator	elePt_2_variableBinning_denominator",
        "effic_eleEta_2_variableBinning       'efficiency vs sub-leading electron eta; electron eta ; efficiency' eleEta_2_variableBinning_numerator       eleEta_2_variableBinning_denominator",
        "effic_elePtEta_2	'efficiency vs sub-leading electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_2_numerator       elePtEta_2_denominator",
        "effic_eleEtaPhi_2	 'efficiency vs sub-leading electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_2_numerator       eleEtaPhi_2_denominator",
        "effic_elePt_3       'efficiency vs trailing electron pt; electron pt [GeV]; efficiency' elePt_3_numerator       elePt_3_denominator",
        "effic_eleEta_3       'efficiency vs trailing electron eta; electron eta ; efficiency' eleEta_3_numerator       eleEta_3_denominator",
        "effic_elePhi_3       'efficiency vs trailing electron phi; electron phi ; efficiency' elePhi_3_numerator       elePhi_3_denominator",
        "effic_elePt_3_variableBinning       'efficiency vs trailing electron pt; electron pt [GeV]; efficiency' elePt_3_variableBinning_numerator       elePt_3_variableBinning_denominator",
        "effic_eleEta_3_variableBinning       'efficiency vs trailing electron eta; electron eta ; efficiency' eleEta_3_variableBinning_numerator       eleEta_3_variableBinning_denominator",
        "effic_elePtEta_3       'efficiency vs trailing electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_3_numerator       elePtEta_3_denominator",
        "effic_eleEtaPhi_3       'efficiency vs trailing electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_3_numerator       eleEtaPhi_3_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
    	"effic_LeadElectronPt_vs_LS 'Electron p_T efficiency vs LS; LS; Electron p_T efficiency' eleVsLS_numerator eleVsLS_denominator"
    ),
)

################################Triple Muon##########################
triplemu12mu10mu5 = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/TriLepton/HLT_TripleMu_12_10_5/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/TriLepton/HLT_TripleMu_12_10_5/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPt_1       'efficiency vs leading muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs leading muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs leading muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs leading muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs leading muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPtEta_1       'efficiency vs leading muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator	muPtEta_1_denominator",
        "effic_muEtaPhi_1	'efficiency vs leading muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator	 muEtaPhi_1_denominator",
        "effic_muPt_2       'efficiency vs sub-leading muon pt; muon pt [GeV]; efficiency' muPt_2_numerator       muPt_2_denominator",
        "effic_muEta_2       'efficiency vs sub-leading muon eta; muon eta ; efficiency' muEta_2_numerator       muEta_2_denominator",
        "effic_muPhi_2       'efficiency vs sub-leading muon phi; muon phi ; efficiency' muPhi_2_numerator       muPhi_2_denominator",
        "effic_muPt_2_variableBinning       'efficiency vs sub-leading muon pt; muon pt [GeV]; efficiency' muPt_2_variableBinning_numerator       muPt_2_variableBinning_denominator",
        "effic_muEta_2_variableBinning       'efficiency vs sub-leading muon eta; muon eta ; efficiency' muEta_2_variableBinning_numerator       muEta_2_variableBinning_denominator",
        "effic_muPtEta_2       'efficiency vs sub-leading muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_2_numerator	muPtEta_2_denominator",
        "effic_muEtaPhi_2	'efficiency vs sub-leading muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_2_numerator	 muEtaPhi_2_denominator",
        "effic_muPt_3       'efficiency vs trailing muon pt; muon pt [GeV]; efficiency' muPt_3_numerator       muPt_3_denominator",
        "effic_muEta_3       'efficiency vs trailing muon eta; muon eta ; efficiency' muEta_3_numerator       muEta_3_denominator",
        "effic_muPhi_3       'efficiency vs trailing muon phi; muon phi ; efficiency' muPhi_3_numerator       muPhi_3_denominator",
        "effic_muPt_3_variableBinning       'efficiency vs trailing muon pt; muon pt [GeV]; efficiency' muPt_3_variableBinning_numerator       muPt_3_variableBinning_denominator",
        "effic_muEta_3_variableBinning       'efficiency vs trailing muon eta; muon eta ; efficiency' muEta_3_variableBinning_numerator       muEta_3_variableBinning_denominator",
        "effic_muPtEta_3       'efficiency vs trailing muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_3_numerator	muPtEta_3_denominator",
        "effic_muEtaPhi_3	'efficiency vs trailing muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_3_numerator	 muEtaPhi_3_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_LeadMuonPt_vs_LS 'Muon p_T efficiency vs LS; LS; Muon p_T efficiency' muVsLS_numerator muVsLS_denominator"
    ),
)

triplemu10mu5mu5DZ = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/TriLepton/HLT_TripleM_10_5_5_DZ/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/TriLepton/HLT_TripleM_10_5_5_DZ/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPt_1       'efficiency vs leading muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs leading muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs leading muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs leading muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs leading muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPtEta_1       'efficiency vs leading muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator	muPtEta_1_denominator",
        "effic_muEtaPhi_1	'efficiency vs leading muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator	 muEtaPhi_1_denominator",
        "effic_muPt_2       'efficiency vs sub-leading muon pt; muon pt [GeV]; efficiency' muPt_2_numerator       muPt_2_denominator",
        "effic_muEta_2       'efficiency vs sub-leading muon eta; muon eta ; efficiency' muEta_2_numerator       muEta_2_denominator",
        "effic_muPhi_2       'efficiency vs sub-leading muon phi; muon phi ; efficiency' muPhi_2_numerator       muPhi_2_denominator",
        "effic_muPt_2_variableBinning       'efficiency vs sub-leading muon pt; muon pt [GeV]; efficiency' muPt_2_variableBinning_numerator       muPt_2_variableBinning_denominator",
        "effic_muEta_2_variableBinning       'efficiency vs sub-leading muon eta; muon eta ; efficiency' muEta_2_variableBinning_numerator       muEta_2_variableBinning_denominator",
        "effic_muPtEta_2       'efficiency vs sub-leading muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_2_numerator	muPtEta_2_denominator",
        "effic_muEtaPhi_2	'efficiency vs sub-leading muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_2_numerator	 muEtaPhi_2_denominator",
        "effic_muPt_3       'efficiency vs trailing muon pt; muon pt [GeV]; efficiency' muPt_3_numerator       muPt_3_denominator",
        "effic_muEta_3       'efficiency vs trailing muon eta; muon eta ; efficiency' muEta_3_numerator       muEta_3_denominator",
        "effic_muPhi_3       'efficiency vs trailing muon phi; muon phi ; efficiency' muPhi_3_numerator       muPhi_3_denominator",
        "effic_muPt_3_variableBinning       'efficiency vs trailing muon pt; muon pt [GeV]; efficiency' muPt_3_variableBinning_numerator       muPt_3_variableBinning_denominator",
        "effic_muEta_3_variableBinning       'efficiency vs trailing muon eta; muon eta ; efficiency' muEta_3_variableBinning_numerator       muEta_3_variableBinning_denominator",
        "effic_muPtEta_3       'efficiency vs trailing muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_3_numerator	muPtEta_3_denominator",
        "effic_muEtaPhi_3	'efficiency vs trailing muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_3_numerator	 muEtaPhi_3_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_LeadMuonPt_vs_LS 'Muon p_T efficiency vs LS; LS; Muon p_T efficiency' muVsLS_numerator muVsLS_denominator"
    ),
)

#############################Double Mu + Single Ele######################################
dimu9ele9caloIdLTrackIdLdz_effmu = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/muLeg/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/muLeg/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPt_1       'efficiency vs leading muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs leading muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs leading muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs leading muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs leading muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPtEta_1       'efficiency vs leading muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator	muPtEta_1_denominator",
        "effic_muEtaPhi_1      'efficiency vs leading muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator	muEtaPhi_1_denominator",
        "effic_muPt_2       'efficiency vs sub-leading muon pt; muon pt [GeV]; efficiency' muPt_2_numerator       muPt_2_denominator",
        "effic_muEta_2       'efficiency vs sub-leading muon eta; muon eta ; efficiency' muEta_2_numerator       muEta_2_denominator",
        "effic_muPhi_2       'efficiency vs sub-leading muon phi; muon phi ; efficiency' muPhi_2_numerator       muPhi_2_denominator",
        "effic_muPt_2_variableBinning       'efficiency vs sub-leading muon pt; muon pt [GeV]; efficiency' muPt_2_variableBinning_numerator       muPt_2_variableBinning_denominator",
        "effic_muEta_2_variableBinning       'efficiency vs sub-leading muon eta; muon eta ; efficiency' muEta_2_variableBinning_numerator       muEta_2_variableBinning_denominator",
        "effic_muPtEta_2       'efficiency vs sub-leading muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_2_numerator	muPtEta_2_denominator",
        "effic_muEtaPhi_2      'efficiency vs sub-leading muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_2_numerator	muEtaPhi_2_denominator",
    ),
)

dimu9ele9caloIdLTrackIdLdz_effele = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/eleLeg/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/eleLeg/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_elePt_1       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator       elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_elePtEta_1       'efficiency vs electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
    ),
)

dimu9ele9caloIdLTrackIdLdz_effdz = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/dzMon/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/dzMon/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPt_1       'efficiency vs leading muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs leading muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs leading muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs leading muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs leading muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPtEta_1       'efficiency vs leading muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator	muPtEta_1_denominator",
        "effic_muEtaPhi_1      'efficiency vs leading muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator	muEtaPhi_1_denominator",
        "effic_muPt_2       'efficiency vs sub-leading muon pt; muon pt [GeV]; efficiency' muPt_2_numerator       muPt_2_denominator",
        "effic_muEta_2       'efficiency vs sub-leading muon eta; muon eta ; efficiency' muEta_2_numerator       muEta_2_denominator",
        "effic_muPhi_2       'efficiency vs sub-leading muon phi; muon phi ; efficiency' muPhi_2_numerator       muPhi_2_denominator",
        "effic_muPt_2_variableBinning       'efficiency vs sub-leading muon pt; muon pt [GeV]; efficiency' muPt_2_variableBinning_numerator       muPt_2_variableBinning_denominator",
        "effic_muEta_2_variableBinning       'efficiency vs sub-leading muon eta; muon eta ; efficiency' muEta_2_variableBinning_numerator       muEta_2_variableBinning_denominator",
        "effic_muPtEta_2       'efficiency vs sub-leading muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_2_numerator	muPtEta_2_denominator",
        "effic_muEtaPhi_2      'efficiency vs sub-leading muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_2_numerator	muEtaPhi_2_denominator",
        "effic_elePt_1       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_numerator       elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator       elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_elePtEta_1       'efficiency vs electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1       'efficiency vs electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
    ),
)

######Double Electron + Single Muon######
mu8diEle12CaloIdLTrackIdL_effele = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/eleLeg/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/eleLeg/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_elePt_1       'efficiency vs leading electron pt; electron pt [GeV]; efficiency' elePt_1_numerator	elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs leading electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs leading electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs leading electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator	elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs leading electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_elePtEta_1	'efficiency vs leading electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1	 'efficiency vs leading electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
        "effic_elePt_2       'efficiency vs sub-leading electron pt; electron pt [GeV]; efficiency' elePt_2_numerator	elePt_2_denominator",
        "effic_eleEta_2       'efficiency vs sub-leading electron eta; electron eta ; efficiency' eleEta_2_numerator       eleEta_2_denominator",
        "effic_elePhi_2       'efficiency vs sub-leading electron phi; electron phi ; efficiency' elePhi_2_numerator       elePhi_2_denominator",
        "effic_elePt_2_variableBinning       'efficiency vs sub-leading electron pt; electron pt [GeV]; efficiency' elePt_2_variableBinning_numerator	elePt_2_variableBinning_denominator",
        "effic_eleEta_2_variableBinning       'efficiency vs sub-leading electron eta; electron eta ; efficiency' eleEta_2_variableBinning_numerator       eleEta_2_variableBinning_denominator",
        "effic_elePtEta_2	'efficiency vs sub-leading electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_2_numerator       elePtEta_2_denominator",
        "effic_eleEtaPhi_2	 'efficiency vs sub-leading electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_2_numerator       eleEtaPhi_2_denominator",
    ),
)
mu8diEle12CaloIdLTrackIdL_effmu = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/muLeg/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/muLeg/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPtEta_1       'efficiency vs muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator       muPtEta_1_denominator",
        "effic_muEtaPhi_1       'efficiency vs muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator       muEtaPhi_1_denominator",
    ),
)

mu8diEle12CaloIdLTrackIdL_effdz = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/dzMon/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/dzMon/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_elePt_1       'efficiency vs leading electron pt; electron pt [GeV]; efficiency' elePt_1_numerator	elePt_1_denominator",
        "effic_eleEta_1       'efficiency vs leading electron eta; electron eta ; efficiency' eleEta_1_numerator       eleEta_1_denominator",
        "effic_elePhi_1       'efficiency vs leading electron phi; electron phi ; efficiency' elePhi_1_numerator       elePhi_1_denominator",
        "effic_elePt_1_variableBinning       'efficiency vs leading electron pt; electron pt [GeV]; efficiency' elePt_1_variableBinning_numerator	elePt_1_variableBinning_denominator",
        "effic_eleEta_1_variableBinning       'efficiency vs leading electron eta; electron eta ; efficiency' eleEta_1_variableBinning_numerator       eleEta_1_variableBinning_denominator",
        "effic_elePtEta_1	'efficiency vs leading electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_1_numerator       elePtEta_1_denominator",
        "effic_eleEtaPhi_1	 'efficiency vs leading electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_1_numerator       eleEtaPhi_1_denominator",
        "effic_elePt_2       'efficiency vs sub-leading electron pt; electron pt [GeV]; efficiency' elePt_2_numerator	elePt_2_denominator",
        "effic_eleEta_2       'efficiency vs sub-leading electron eta; electron eta ; efficiency' eleEta_2_numerator       eleEta_2_denominator",
        "effic_elePhi_2       'efficiency vs sub-leading electron phi; electron phi ; efficiency' elePhi_2_numerator       elePhi_2_denominator",
        "effic_elePt_2_variableBinning       'efficiency vs sub-leading electron pt; electron pt [GeV]; efficiency' elePt_2_variableBinning_numerator	elePt_2_variableBinning_denominator",
        "effic_eleEta_2_variableBinning       'efficiency vs sub-leading electron eta; electron eta ; efficiency' eleEta_2_variableBinning_numerator       eleEta_2_variableBinning_denominator",
        "effic_elePtEta_2	'efficiency vs sub-leading electron pt-#eta; electron pt [GeV]; electron #eta' elePtEta_2_numerator       elePtEta_2_denominator",
        "effic_eleEtaPhi_2	 'efficiency vs sub-leading electron #eta-#phi; electron #eta ; electron #phi' eleEtaPhi_2_numerator       eleEtaPhi_2_denominator",
        "effic_muPt_1       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_numerator       muPt_1_denominator",
        "effic_muEta_1       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_numerator       muEta_1_denominator",
        "effic_muPhi_1       'efficiency vs muon phi; muon phi ; efficiency' muPhi_1_numerator       muPhi_1_denominator",
        "effic_muPt_1_variableBinning       'efficiency vs muon pt; muon pt [GeV]; efficiency' muPt_1_variableBinning_numerator       muPt_1_variableBinning_denominator",
        "effic_muEta_1_variableBinning       'efficiency vs muon eta; muon eta ; efficiency' muEta_1_variableBinning_numerator       muEta_1_variableBinning_denominator",
        "effic_muPtEta_1       'efficiency vs muon pt-#eta; muon pt [GeV]; muon #eta' muPtEta_1_numerator       muPtEta_1_denominator",
        "effic_muEtaPhi_1       'efficiency vs muon #eta-#phi; muon #eta ; muon #phi' muEtaPhi_1_numerator       muEtaPhi_1_denominator",
    ),
)
### mia: FOCA D'OVATTA !
diphotonEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/photon/HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v",
                                           "HLT/photon/HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v",
                                           "HLT/photon/HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_Mass55_v"),
                                    #subDirs        = cms.untracked.vstring("HLT/Higgs/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "eff_diphoton_pt       'efficiency vs lead pt;             Photon pt [GeV]; efficiency'     photon_pt_numerator          photon_pt_denominator",
        "eff_diphoton_variable 'efficiency vs lead pt;             Photon pt [GeV]; efficiency'     photon_pt_variable_numerator photon_pt_variable_denominator",
        "eff_diphoton_eta      'efficiency vs lead eta;            Photon eta; efficiency'          photon_eta_numerator         photon_eta_denominator",
        "eff_diphoton_subpt    'efficiency vs sublead pt;          Photon subpt [GeV]; efficiency'  subphoton_pt_numerator       subphoton_pt_denominator",
        "eff_diphoton_subeta   'efficiency vs sublead eta;         Photon subeta; efficiency'       subphoton_eta_numerator      subphoton_eta_denominator",
        "eff_diphoton_mass     'efficiency vs diphoton mass;       Diphoton mass; efficiency'       diphoton_mass_numerator      diphoton_mass_denominator",
        "eff_photon_phi        'efficiency vs lead phi;            Photon phi [rad]; efficiency'    photon_phi_numerator         photon_phi_denominator",
        "eff_photon_subphi     'efficiency vs sublead phi;         Photon subphi [rad]; efficiency' subphoton_phi_numerator      subphoton_phi_denominator",
        "eff_photonr9          'efficiency vs r9;                  Photon r9; efficiency'           photon_r9_numerator          photon_r9_denominator",
        "eff_photonhoE         'efficiency vs hoE;                 Photon hoE; efficiency'          photon_hoE_numerator         photon_hoE_denominator",
        "eff_photonEtaPhi      'Photon phi;                        Photon eta; efficiency'          photon_etaphi_numerator      photon_etaphi_denominator",
        "eff_photon_subr9      'efficiency vs sublead r9;          Photon subr9; efficiency'        subphoton_r9_numerator       subphoton_r9_denominator",
        "eff_photon_subhoE     'efficiency vs sublead hoE;         Photon subhoE; efficiency'       subphoton_hoE_numerator      subphoton_hoE_denominator",
        "eff_photon_subEtaPhi  'Photon sublead phi;                Photon sublead eta; efficiency'  subphoton_etaphi_numerator   subphoton_etaphi_denominator",

    ),
    efficiencyProfile = cms.untracked.vstring(
        "eff_photon_vs_LS 'Photon pt efficiency vs LS; LS' photonVsLS_numerator photonVsLS_denominator"
    ),
)

VBFEfficiency = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/VBFHbb/*"),
    subDirs        = cms.untracked.vstring("HLT/HIG/VBFHbb/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPhi_1      'efficiency vs 1st jet phi; jet phi ; efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        "effic_jetPhi_2      'efficiency vs 2nd jet phi; jet phi ; efficiency'    jetPhi_2_numerator      jetPhi_2_denominator",
        "effic_jetPhi_3      'efficiency vs 3rd jet phi; jet phi ; efficiency'    jetPhi_3_numerator      jetPhi_3_denominator",
        "effic_jetPhi_4      'efficiency vs 4th jet phi; jet phi ; efficiency'    jetPhi_4_numerator      jetPhi_4_denominator",
        #
        "effic_bjetPhi_1     'efficiency vs 1st b-jet phi; bjet phi ; efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetCSV_1     'efficiency vs 1st b-jet csv; bjet CSV; efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        #
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet phi; bjet phi ; efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet csv; bjet CSV; efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",
        #
        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet pt; jet pt [GeV]; efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs 2nd jet pt; jet pt [GeV]; efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetPt_3_variableBinning       'efficiency vs 3rd jet pt; jet pt [GeV]; efficiency' jetPt_3_variableBinning_numerator       jetPt_3_variableBinning_denominator",
        "effic_jetPt_4_variableBinning       'efficiency vs 4th jet pt; jet pt [GeV]; efficiency' jetPt_4_variableBinning_numerator       jetPt_4_variableBinning_denominator",
        #
        "effic_jetEta_1_variableBinning       'efficiency vs 1st jet eta; jet eta ; efficiency' jetEta_1_variableBinning_numerator       jetEta_1_variableBinning_denominator",
        "effic_jetEta_2_variableBinning       'efficiency vs 2nd jet eta; jet eta ; efficiency' jetEta_2_variableBinning_numerator       jetEta_2_variableBinning_denominator",
        "effic_jetEta_3_variableBinning       'efficiency vs 3rd jet eta; jet eta ; efficiency' jetEta_3_variableBinning_numerator       jetEta_3_variableBinning_denominator",
        "effic_jetEta_4_variableBinning       'efficiency vs 4th jet eta; jet eta ; efficiency' jetEta_4_variableBinning_numerator       jetEta_4_variableBinning_denominator",
        #
        "effic_bjetPt_1_variableBinning   'efficiency vs 1st b-jet pt; bjet pt [GeV]; efficiency' bjetPt_1_variableBinning_numerator   bjetPt_1_variableBinning_denominator",
        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet eta; bjet eta ; efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        #
        "effic_bjetPt_2_variableBinning   'efficiency vs 2nd b-jet pt; bjet pt [GeV]; efficiency' bjetPt_2_variableBinning_numerator   bjetPt_2_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet eta; bjet eta ; efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",
        #
        "effic_jetMulti       'efficiency vs jet multiplicity; jet multiplicity; efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity; bjet multiplicity; efficiency' bjetMulti_numerator   bjetMulti_denominator",
        #
        "effic_jetPtEta_1     'efficiency vs 1st jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetPtEta_2     'efficiency vs 2nd jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetPtEta_3     'efficiency vs 3rd jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_3_numerator       jetPtEta_3_denominator",
        "effic_jetPtEta_4     'efficiency vs 4th jet pt-#eta; jet pt [GeV]; jet #eta' jetPtEta_4_numerator       jetPtEta_4_denominator",
        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetEtaPhi_2    'efficiency vs 2nd jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_3    'efficiency vs 3rd jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_3_numerator       jetEtaPhi_3_denominator",
        "effic_jetEtaPhi_4    'efficiency vs 4th jet #eta-#phi; jet #eta ; jet #phi' jetEtaPhi_4_numerator       jetEtaPhi_4_denominator",
        #
        "effic_bjetPtEta_1    'efficiency vs 1st b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        #
        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        #
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet pt-#eta; jet pt [GeV]; bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",
        #
        "effic_bjetEtaPhi_2    'efficiency vs 2nd b-jet #eta-#phi; bjet #eta ; bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",
        ),
)



higgsClient = cms.Sequence(
    diphotonEfficiency
  + vbfmetClient
  + vbftauClient
  + ele23Ele12CaloIdLTrackIdLIsoVL_effdz
  + dimu9ele9caloIdLTrackIdLdz_effmu
  + dimu9ele9caloIdLTrackIdLdz_effele
  + dimu9ele9caloIdLTrackIdLdz_effdz
  + mu8diEle12CaloIdLTrackIdL_effmu
  + mu8diEle12CaloIdLTrackIdL_effele
  + mu8diEle12CaloIdLTrackIdL_effdz
  + ele16ele12ele8caloIdLTrackIdL
  + triplemu12mu10mu5
  + triplemu10mu5mu5DZ
  + muEleDz_effmu
  + muEleDz_effele
#  + mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_effele
#  + mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_effmu
#  + mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_effele
#  + mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_effmu
  + metbtagEfficiency_met
  + metbtagEfficiency_btag
  + VBFEfficiency
  + mssmHbbBtagTriggerEfficiency 
  + mssmHbbHLTEfficiency 
  + hmesongammaEfficiency
)
