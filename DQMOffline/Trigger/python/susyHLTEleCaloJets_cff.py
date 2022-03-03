import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SusyMonitor_cfi import hltSUSYmonitoring

#This is added by Pablo in order to monitor the auxiliary paths for electron fake rate calculation
susyEle8CaloJet30_jet = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele8CaloJet30/JetMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>50 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,50,60,80,120,200,400],
                     elePtBinning2D = [0,50,70,120,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele35_WPTight_Gsf_v*', 'HLT_Ele38_WPTight_Gsf_v*', 'HLT_Ele40_WPTight_Gsf_v*'])
)
susyEle8CaloJet30_ele = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele8CaloJet30/ElectronMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>10 & abs(eta)<2.4',
    jetSelection = 'pt>80 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,10,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,10,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,50,60,80,120,200,400],
                     jetPtBinning2D = [0,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFJet60_v*'])
)
susyEle8CaloJet30_all = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele8CaloJet30/GlobalMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>10 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,10,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,10,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v*'])
    #denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu24_v*'])
)
susyEle8CaloIdMJet30_jet = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele8CaloIdMJet30/JetMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>50 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,50,60,80,120,200,400],
                     elePtBinning2D = [0,50,70,120,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele35_WPTight_Gsf_v*', 'HLT_Ele38_WPTight_Gsf_v*', 'HLT_Ele40_WPTight_Gsf_v*'])
)
susyEle8CaloIdMJet30_ele = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele8CaloIdMJet30/ElectronMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>10 & abs(eta)<2.4',
    jetSelection = 'pt>80 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,10,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,10,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,50,60,80,120,200,400],
                     jetPtBinning2D = [0,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFJet60_v*'])
)
susyEle8CaloIdMJet30_all = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele8CaloIdMJet30/GlobalMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>10 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,10,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,10,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v*'])
    #denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu24_v*'])
)
susyEle12CaloJet30_jet = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele12CaloJet30/JetMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>50 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,50,60,80,120,200,400],
                     elePtBinning2D = [0,50,70,120,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele35_WPTight_Gsf_v*', 'HLT_Ele38_WPTight_Gsf_v*', 'HLT_Ele40_WPTight_Gsf_v*'])
)
susyEle12CaloJet30_ele = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele12CaloJet30/ElectronMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>14 & abs(eta)<2.4',
    jetSelection = 'pt>80 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,12,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,12,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,50,60,80,120,200,400],
                     jetPtBinning2D = [0,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFJet60_v*'])
)
susyEle12CaloJet30_all = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele12CaloJet30/GlobalMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>14 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,12,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,12,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v*'])
    #denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu24_v*'])
)
susyEle17CaloIdMJet30_jet = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele17CaloIdMJet30/JetMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>50 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,50,60,80,120,200,400],
                     elePtBinning2D = [0,50,70,120,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele35_WPTight_Gsf_v*', 'HLT_Ele38_WPTight_Gsf_v*', 'HLT_Ele40_WPTight_Gsf_v*'])
)
susyEle17CaloIdMJet30_ele = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele17CaloIdMJet30/ElectronMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>19 & abs(eta)<2.4',
    jetSelection = 'pt>80 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,19,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,19,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,50,60,80,120,200,400],
                     jetPtBinning2D = [0,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFJet60_v*'])
)
susyEle17CaloIdMJet30_all = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele17CaloIdMJet30/GlobalMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>19 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,19,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,19,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v*']),
    #denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu24_v*'])
)
susyEle23CaloJet30_jet = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele23CaloJet30/JetMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>50 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,50,60,80,120,200,400],
                     elePtBinning2D = [0,50,70,120,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele35_WPTight_Gsf_v*', 'HLT_Ele38_WPTight_Gsf_v*', 'HLT_Ele40_WPTight_Gsf_v*'])
)
susyEle23CaloJet30_ele = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele23CaloJet30/ElectronMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>25 & abs(eta)<2.4',
    jetSelection = 'pt>80 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,50,60,80,120,200,400],
                     jetPtBinning2D = [0,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFJet60_v*'])
)
susyEle23CaloJet30_all = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele23CaloJet30/GlobalMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>14 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,12,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,12,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v*']),
    #denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu24_v*'])
)
susyEle23CaloIdMJet30_jet = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele23CaloIdMJet30/JetMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>50 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,50,60,80,120,200,400],
                     elePtBinning2D = [0,50,70,120,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele35_WPTight_Gsf_v*', 'HLT_Ele38_WPTight_Gsf_v*', 'HLT_Ele40_WPTight_Gsf_v*'])
)
susyEle23CaloIdMJet30_ele = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele23CaloIdMJet30/ElectronMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>25 & abs(eta)<2.4',
    jetSelection = 'pt>80 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,50,60,80,120,200,400],
                     jetPtBinning2D = [0,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFJet60_v*'])
)
susyEle23CaloIdMJet30_all = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/Ele23CaloIdMJet30/GlobalMonitor',
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>14 & abs(eta)<2.4',
    jetSelection = 'pt>35 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,12,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,12,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v*']),
    #denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu24_v*'])
)
susyHLTEleCaloJets = cms.Sequence(
    susyEle8CaloJet30_ele
  + susyEle8CaloJet30_jet
  + susyEle8CaloJet30_all
  + susyEle12CaloJet30_ele
  + susyEle12CaloJet30_jet
  + susyEle12CaloJet30_all
  + susyEle23CaloJet30_ele
  + susyEle23CaloJet30_jet
  + susyEle23CaloJet30_all
  + susyEle8CaloIdMJet30_ele
  + susyEle8CaloIdMJet30_jet
  + susyEle8CaloIdMJet30_all
  + susyEle17CaloIdMJet30_ele
  + susyEle17CaloIdMJet30_jet
  + susyEle17CaloIdMJet30_all
  + susyEle23CaloIdMJet30_ele
  + susyEle23CaloIdMJet30_jet
  + susyEle23CaloIdMJet30_all
)
