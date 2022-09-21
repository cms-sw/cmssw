import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.PhotonMonitor_cff import *
from DQMOffline.Trigger.VBFMETMonitor_cff import *
from DQMOffline.Trigger.HMesonGammaMonitor_cff import *
from DQMOffline.Trigger.METMonitor_cfi import hltMETmonitoring
from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring
from DQMOffline.Trigger.VBFTauMonitor_cff import *
from DQMOffline.Trigger.MssmHbbBtagTriggerMonitor_cff import *
from DQMOffline.Trigger.MssmHbbMonitoring_cff import *
from DQMOffline.Trigger.HiggsMonitoring_cfi import hltHIGmonitoring
from DQMOffline.Trigger.BTaggingMonitor_cfi import hltBTVmonitoring

# HLT_PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1 MET monitoring
PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_METmonitoring = hltMETmonitoring.clone(
#FolderName = 'HLT/Higgs/PFMET100_BTag/'
    FolderName = 'HLT/HIG/PFMET100_BTag/',
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_v"])
)

# HLT_PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1 MET monitoring
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_METmonitoring = hltMETmonitoring.clone(
    #FolderName = 'HLT/Higgs/PFMET110_BTag/',
    FolderName = 'HLT/HIG/PFMET110_BTag/',
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_v"])
)

# HLT_PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1 b-tag monitoring
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring = hltTOPmonitoring.clone(
    #FolderName= 'HLT/Higgs/PFMET110_BTag/',
    FolderName= 'HLT/HIG/PFMET110_BTag/',
    # Selection
    leptJetDeltaRmin = 0.0,
    njets            = 1,
    jetSelection     = 'pt>30 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 0,
    nbjets           = 1,
    bjetSelection    = 'pt>30 & abs(eta)<2.4',
    workingpoint     = 0.8484, # Medium
    # Binning                                                                                                          
    histoPSet = dict(
        htPSet = dict(nbins=50, xmin=0.0, xmax=1000),
        jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400],
        HTBinning    = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900],
        metBinning = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900]
    ),
    # Triggers                                                                                                         
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_v']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFMET110_PFMHT110_IDTight_v'])
)

# HLT_PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1 MET monitoring
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_METmonitoring = hltMETmonitoring.clone(
    #FolderName = 'HLT/Higgs/PFMET120_BTag/'
    FolderName = 'HLT/HIG/PFMET120_BTag/',
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_v"])
)

# HLT_PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1 b-tag monitoring
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring = hltTOPmonitoring.clone(
    #FolderName= 'HLT/Higgs/PFMET120_BTag/',
    FolderName= 'HLT/HIG/PFMET120_BTag/',
    # Selection
    leptJetDeltaRmin = 0.0,
    njets            = 1,
    jetSelection     = 'pt>30 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 0,
    nbjets           = 1,
    bjetSelection    = 'pt>30 & abs(eta)<2.4',
    workingpoint     = 0.8484, # Medium
    # Binning                                                                                                         
    histoPSet = dict(
        htPSet = dict(nbins=50, xmin=0.0, xmax=1000 ),
        jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400],
        HTBinning    = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900],
        metBinning = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900]
    ),
    # Triggers                                                                                                        
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_v']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFMET120_PFMHT120_IDTight_v'])	
)


# HLT_PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1 MET monitoring
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_METmonitoring = hltMETmonitoring.clone(
    #FolderName = 'HLT/Higgs/PFMET130_BTag/',
    FolderName = 'HLT/HIG/PFMET130_BTag/',
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_v"])
)

# HLT_PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1 b-tag monitoring
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring = hltTOPmonitoring.clone(
    #FolderName= 'HLT/Higgs/PFMET130_BTag/'
    FolderName= 'HLT/HIG/PFMET130_BTag/',
    # Selection
    leptJetDeltaRmin = 0.0,
    njets            = 1,
    jetSelection     = 'pt>30 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 0,
    nbjets           = 1,
    bjetSelection    = 'pt>30 & abs(eta)<2.4',
    workingpoint     = 0.8484, # Medium
    # Binning                                                                                                         
    histoPSet = dict(
        htPSet = dict(nbins=50, xmin=0.0, xmax=1000 ),
        jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,130,200,400],
        HTBinning    = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900],
        metBinning = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900]
    ),
    # Triggers                                                                                                        
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_v']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFMET130_PFMHT130_IDTight_v'])
)

# HLT_PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1 MET monitoring
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_METmonitoring = hltMETmonitoring.clone(
    #FolderName = 'HLT/Higgs/PFMET140_BTag/'
    FolderName = 'HLT/HIG/PFMET140_BTag/',
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_v"])
)

# HLT_PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1 b-tag monitoring
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring = hltTOPmonitoring.clone(
    #FolderName= 'HLT/Higgs/PFMET140_BTag/',
    FolderName= 'HLT/HIG/PFMET140_BTag/',
    # Selection
    leptJetDeltaRmin = 0.0,
    njets            = 1,
    jetSelection     = 'pt>30 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 0,
    nbjets           = 1,
    bjetSelection    = 'pt>30 & abs(eta)<2.4',
    workingpoint     = 0.8484, # Medium
    # Binning                                                                                                         
    histoPSet = dict(
        htPSet = dict(nbins=50, xmin=0.0, xmax=1000 ),
        jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,140,200,400],
        HTBinning    = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900],
        metBinning = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900]
    ),
    # Triggers                                                                                                        
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_v']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFMET140_PFMHT140_IDTight_v'])
)

#######for HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ####
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon = hltHIGmonitoring.clone(
    nelectrons = 2,
    #FolderName = 'HLT/Higgs/DiLepton/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
    FolderName = 'HLT/HIG/DiLepton/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v*"])
)

##############################DiLepton cross triggers######################################################
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg = hltHIGmonitoring.clone(
    nmuons = 1,
    nelectrons = 1,
    #FolderName = 'HLT/Higgs/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg',
    FolderName = 'HLT/HIG/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu20_v*",
                                                  "HLT_IsoMu24_eta2p1_v*",
                                                  "HLT_IsoMu24_v*",
                                                  "HLT_IsoMu27_v*",
                                                  "HLT_IsoMu20_v*"])
)

mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg = hltHIGmonitoring.clone(
    nmuons = 1,
    nelectrons = 1,
    #FolderName = 'HLT/Higgs/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/muLeg',
    FolderName = 'HLT/HIG/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/muLeg',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Ele27_WPTight_Gsf_v*",
                                                  "HLT_Ele35_WPTight_Gsf_v*"])
)


#####HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v#####
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg = hltHIGmonitoring.clone(
    nmuons = 1,
    nelectrons = 1,
    #FolderName = 'HLT/Higgs/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg',
    FolderName = 'HLT/HIG/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*"]), #        
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu20_v*",
                                                  "HLT_IsoMu24_eta2p1_v*",
                                                  "HLT_IsoMu24_v*",
                                                  "HLT_IsoMu27_v*",
                                                  "HLT_IsoMu20_v*"])
)


mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg = hltHIGmonitoring.clone(
    nmuons = 1,
    nelectrons = 1,
    #FolderName = 'HLT/Higgs/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/muLeg',
    FolderName = 'HLT/HIG/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/muLeg',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Ele27_WPTight_Gsf_v*",
                                                  "HLT_Ele35_WPTight_Gsf_v*"])
)

###############################same flavour trilepton monitor####################################
########TripleMuon########
higgsTrimumon = hltHIGmonitoring.clone(
    #FolderName = 'HLT/Higgs/TriLepton/HLT_TripleMu_12_10_5/',
    FolderName = 'HLT/HIG/TriLepton/HLT_TripleMu_12_10_5/',
    nmuons = 3,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_TripleMu_12_10_5_v*"]), #                                      
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*"])
)

higgsTrimu10_5_5_dz_mon = hltHIGmonitoring.clone(
    #FolderName = 'HLT/Higgs/TriLepton/HLT_TripleM_10_5_5_DZ/',
    FolderName = 'HLT/HIG/TriLepton/HLT_TripleM_10_5_5_DZ/',
    nmuons = 3,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_TripleMu_10_5_5_DZ_v*"]), #                                    
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*"])
)

#######TripleElectron####
higgsTrielemon = hltHIGmonitoring.clone(
    #FolderName = 'HLT/Higgs/TriLepton/HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL/',
    FolderName = 'HLT/HIG/TriLepton/HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL/',
    nelectrons = 3,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v*"]), #                     
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*"])
)

###############################cross flavour trilepton monitor####################################
#########DiMuon+Single Ele Trigger###################
diMu9Ele9CaloIdLTrackIdL_muleg = hltHIGmonitoring.clone(
#FolderName = 'HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/muLeg',
    FolderName = 'HLT/HIG/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/muLeg',
    nelectrons = 1,
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*",
                                                  "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*"])
)


diMu9Ele9CaloIdLTrackIdL_eleleg = hltHIGmonitoring.clone(
    #FolderName = 'HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/eleLeg',
    FolderName = 'HLT/HIG/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/eleLeg',
    nelectrons = 1,
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*",
                                                  "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*"])
)

##Eff of the HLT with DZ w.ref to non-DZ one
diMu9Ele9CaloIdLTrackIdL_dz = hltHIGmonitoring.clone(
    #FolderName = 'HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/dzMon',
    FolderName = 'HLT/HIG/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/dzMon',
    nelectrons = 1,
    nmuons = 2
)
diMu9Ele9CaloIdLTrackIdL_dz.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ_v*")
diMu9Ele9CaloIdLTrackIdL_dz.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*")

#################DiElectron+Single Muon Trigger##################
mu8diEle12CaloIdLTrackIdL_eleleg = hltHIGmonitoring.clone(
    #FolderName = 'HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/eleLeg',
    FolderName = 'HLT/HIG/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/eleLeg',
    nelectrons = 2,
    nmuons = 1,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*",
                                                  "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*"])
)

mu8diEle12CaloIdLTrackIdL_muleg = hltHIGmonitoring.clone(
    #FolderName = 'HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/muLeg',
    FolderName = 'HLT/HIG/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/muLeg',
    nelectrons = 2,
    nmuons = 1,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*"])

)

##Eff of the HLT with DZ w.ref to non-DZ one
mu8diEle12CaloIdLTrackIdL_dz = hltHIGmonitoring.clone(
    #FolderName = 'HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/dzMon',
    FolderName = 'HLT/HIG/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/dzMon',
    nelectrons = 2,
    nmuons = 1,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*"])
)


##VBF triggers##
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1 = hltTOPmonitoring.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1_v',
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1_v',
    # Selection
    leptJetDeltaRmin = 0.0,
    njets            = 4,
    jetSelection     = 'pt>15 & abs(eta)<4.7',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 0,
    nbjets           = 2,
    bjetSelection    = 'pt>15 & abs(eta)<4.7',
    btagAlgos        = ["pfCombinedMVAV2BJetTags"],
    workingpoint     = -0.715, # Loose
    # Triggers                                                                                                                                                                                                                                                                  
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1_v*']),
    # Binning                                                                                                                                                                                                                                                                   
    #QuadPFJet_BTagCSV_p016_p11_VBF_Mqq240.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )                                                                                                                                                
    histoPSet = dict(
        jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400],
        HTBinning    = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900],
        metBinning = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900],

        lsPSet = dict(
            nbins =  1
        ),

        htPSet = dict(
            nbins =  1 ,
            xmin  =  0 ,
            xmax  =  1
        ),
        csvPSet = dict(
            nbins = 20 ,
            xmin  = 0 ,
            xmax  = 1
        ),
        etaPSet = dict(
            nbins =  1 ,
            xmin  =  0 ,
            xmax  =  1
        ),
        ptPSet = dict(
            nbins = 1,
            xmin  = 0 ,
            xmax  = 1
        )
    )
)

QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1 = QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1_v',                                                          
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1_v',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1_v*'])
)

QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1 = QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1_v',                                                          
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1_v',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1_v*'])
)

QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1 = QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1_v',                                                          
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1_v',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1_v*'])
)

QuadPFJet98_83_71_15_BTagCSV_p013_VBF1 = hltTOPmonitoring.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet98_83_71_15_BTagCSV_p013_VBF2_v',                                                                     
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet98_83_71_15_BTagCSV_p013_VBF2_v',
    # Selection                                                                                                                                       
    leptJetDeltaRmin = 0.0,
    njets            = 4,
    jetSelection     = 'pt>15 & abs(eta)<4.7',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 0,
    nbjets           = 1,
    bjetSelection    = 'pt>15 & abs(eta)<4.7',
    btagAlgos        = ["pfCombinedMVAV2BJetTags"],
    workingpoint     = -0.715, # Loose                                                                                                                
    # Triggers                                                                                                                                        
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet98_83_71_15_BTagCSV_p013_VBF2_v*']),
    # Binning                                                                                                                                         
    #QuadPFJet_BTagCSV_p016_p11_VBF_Mqq240.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )                      
    histoPSet = dict(
        jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400],
        HTBinning    = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900],
        metBinning = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900],

        lsPSet = dict(
            nbins =  1 ,
        ),
        htPSet = dict(
            nbins = 1 ,
            xmin  = 0 ,
            xmax  = 1 ,
        ),
        csvPSet = dict(
            nbins = 20,
            xmin  = 0 ,
            xmax  = 1 ,
        ),
        etaPSet = dict(
            nbins =  1,
            xmin  =  0 ,
            xmax  =  1 ,
        ),
        ptPSet = dict(
            nbins =  1 ,
            xmin  =  0 ,
            xmax  =  1 ,
        )
    )
)

QuadPFJet103_88_75_15_BTagCSV_p013_VBF1 = QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet103_88_75_15_BTagCSV_p013_VBF2_v',
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet103_88_75_15_BTagCSV_p013_VBF2_v',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet103_88_75_15_BTagCSV_p013_VBF2_v*'])
)

QuadPFJet105_88_76_15_BTagCSV_p013_VBF1 = QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet105_88_76_15_BTagCSV_p013_VBF2_v',
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet105_88_76_15_BTagCSV_p013_VBF2_v',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet105_88_76_15_BTagCSV_p013_VBF2_v*'])
)

QuadPFJet111_90_80_15_BTagCSV_p013_VBF1 = QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet111_90_80_15_BTagCSV_p013_VBF2_v',
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet111_90_80_15_BTagCSV_p013_VBF2_v',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet111_90_80_15_BTagCSV_p013_VBF2_v*'])
)

QuadPFJet98_83_71_15 = hltTOPmonitoring.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet98_83_71_15_v',
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet98_83_71_15_v',
    # Selection
    leptJetDeltaRmin = 0.0,
    njets            = 4,
    jetSelection     = 'pt>15 & abs(eta)<4.7',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 0,
    nbjets           = 0,
    bjetSelection    = 'pt>15 & abs(eta)<4.7',
    btagAlgos        = ["pfCombinedMVAV2BJetTags"],
    workingpoint     = -0.715, # Loose
	# Triggers                                                                                                                                        
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet98_83_71_15_v*']),
    # Binning                                                                                                                                         
    #QuadPFJet_BTagCSV_p016_p11_VBF_Mqq240.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )                      
    histoPSet = dict(
    jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400],
        HTBinning    = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900],
        metBinning = [0,20,40,60,80,100,125,150,175,200,300,400,500,700,900],

        lsPSet = dict(
            nbins = 1
        ),
        htPSet = dict(
            nbins = 1 ,
            xmin  = 0 ,
            xmax  = 1
        ),
        csvPSet = dict(
            nbins =  20,
            xmin  =  0 ,
            xmax  =  1
        ),
        etaPSet = dict(
            nbins = 1 ,
            xmin  = 0 ,
            xmax  = 1
        ),
        ptPSet = dict(
            nbins =  1 ,
            xmin  =  0 ,
            xmax  =  1
        )
    )
)

QuadPFJet103_88_75_15 = QuadPFJet98_83_71_15.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet103_88_75_15_v',
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet103_88_75_15_v',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet103_88_75_15_v*'])
)

QuadPFJet105_88_76_15 = QuadPFJet98_83_71_15.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet105_88_76_15_v',
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet105_88_76_15_v',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet105_88_76_15_v*'])
)

QuadPFJet111_90_80_15 = QuadPFJet98_83_71_15.clone(
    #FolderName= 'HLT/Higgs/VBFHbb/HLT_QuadPFJet111_90_80_15_v',
    FolderName= 'HLT/HIG/VBFHbb/HLT_QuadPFJet111_90_80_15_v',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_QuadPFJet111_90_80_15_v*'])
)

###############################Higgs Monitor HLT##############################################
higgsMonitorHLT = cms.Sequence(
### THEY WERE IN EXTRA
    higgsinvHLTJetMETmonitoring
  + higgsHLTDiphotonMonitoring
  + higgstautauHLTVBFmonitoring
  + higgsTrielemon
  + higgsTrimumon
  + higgsTrimu10_5_5_dz_mon
  + ele23Ele12CaloIdLTrackIdLIsoVL_dzmon
  + mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg
  + mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg
  + mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg
  + mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg
  + mu8diEle12CaloIdLTrackIdL_muleg
  + mu8diEle12CaloIdLTrackIdL_eleleg
  + mu8diEle12CaloIdLTrackIdL_dz
  + diMu9Ele9CaloIdLTrackIdL_muleg
  + diMu9Ele9CaloIdLTrackIdL_eleleg
  + diMu9Ele9CaloIdLTrackIdL_dz
  + PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_METmonitoring
  + PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_METmonitoring
  + PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring
  + PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_METmonitoring
  + PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring
  + PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_METmonitoring
  + PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring
  + PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_METmonitoring
  + PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring
  + QuadPFJet98_83_71_15_BTagCSV_p013_VBF1
  + QuadPFJet103_88_75_15_BTagCSV_p013_VBF1
  + QuadPFJet105_88_76_15_BTagCSV_p013_VBF1
  + QuadPFJet111_90_80_15_BTagCSV_p013_VBF1
  + QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1
  + QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1
  + QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1
  + QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1
  + QuadPFJet98_83_71_15
  + QuadPFJet103_88_75_15
  + QuadPFJet105_88_76_15
  + QuadPFJet111_90_80_15	
  + mssmHbbBtagTriggerMonitor 
  + mssmHbbMonitorHLT 
  + HMesonGammamonitoring
)


higHLTDQMSourceExtra = cms.Sequence(
)
