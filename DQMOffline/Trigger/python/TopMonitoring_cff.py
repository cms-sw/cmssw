import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

from Configuration.Eras.Modifier_run2_HLTconditions_2018_cff import run2_HLTconditions_2018
from Configuration.Eras.Modifier_run2_HLTconditions_2017_cff import run2_HLTconditions_2017
from Configuration.Eras.Modifier_run2_HLTconditions_2016_cff import run2_HLTconditions_2016

###
### Ele+Jet
###

topEleJet_jet = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/EleJet/JetMonitor',
    enable2DPlots = False,
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>50 & abs(eta)<2.1',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,50,60,80,120,200,400],
                     elePtBinning2D = [0,50,70,120,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet= dict(hltPaths = ['HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele35_WPTight_Gsf_v*',
                                                  'HLT_Ele38_WPTight_Gsf_v*',
                                                  'HLT_Ele40_WPTight_Gsf_v*',])
)
### ---

topEleJet_ele = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/EleJet/ElectronMonitor',
    enable2DPlots = False,
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>25 & abs(eta)<2.1',
    jetSelection = 'pt>50 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,50,60,80,120,200,400],
                     jetPtBinning2D = [0,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFJet60_v*',
                                                  'HLT_PFJet80_v*',
                                                  'HLT_PFJet140_v*',
                                                  'HLT_PFJet200_v*',
                                                  'HLT_PFJet260_v*',
                                                  'HLT_PFJet320_v*',
                                                  'HLT_PFJet400_v*',
                                                  'HLT_PFJet450_v*',
                                                  'HLT_PFJet500_v*',
                                                  'HLT_PFJet550_v*'])
)
### ---
topEleJet_all = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/EleJet/GlobalMonitor',
    enable2DPlots = False,
    nmuons = 0,
    nelectrons = 1,
    njets = 1,
    eleSelection = 'pt>25 & abs(eta)<2.1',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,30,35,37.5,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,35,40,50,60,80,100,200,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*']),
    #denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu24_v*'])
)
###
### Ele+HT
###

topEleHT_ht = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/EleHT/HTMonitor',
    enable2DPlots = False,
    nmuons = 0,
    nelectrons = 1,
    njets = 2,
    eleSelection = 'pt>50 & abs(eta)<2.1',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    HTcut = 100,
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,50,60,80,120,200,400],
                     elePtBinning2D = [0,50,70,120,200,400],
                     jetPtBinning = [0,30,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,40,60,80,100,200,400],
                     HTBinning  = [0,100,120,140,150,160,175,200,300,400,500,700],
                     HTBinning2D  = [0,100,125,150,175,200,400,700]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele35_WPTight_Gsf_v*',
                                                  'HLT_Ele38_WPTight_Gsf_v*',
                                                  'HLT_Ele40_WPTight_Gsf_v*'])
)
### ---

topEleHT_ele = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/EleHT/ElectronMonitor',
    enable2DPlots = False,
    nmuons = 0,
    nelectrons = 1,
    njets = 2,
    eleSelection = 'pt>25 & abs(eta)<2.1',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    HTcut = 200,
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,30,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,40,60,80,100,200,400],
                     HTBinning  = [0,200,250,300,350,400,500,700],
                     HTBinning2D  = [0,200,250,300,400,500,700]),
    numGenericTriggerEventPSet =dict(hltPaths = ['HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT250_v*',
                                                  'HLT_PFHT370_v*',
                                                  'HLT_PFHT430_v*',
                                                  'HLT_PFHT510_v*',
                                                  'HLT_PFHT590_v*',
                                                  'HLT_PFHT680_v*',
                                                  'HLT_PFHT780_v*',
                                                  'HLT_PFHT890_v*'])
)
### ---

topEleHT_all = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/EleHT/GlobalMonitor',
    enable2DPlots = False,
    nmuons = 0,
    nelectrons = 1,
    njets = 2,
    eleSelection = 'pt>25 & abs(eta)<2.1',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    HTcut = 100,
    histoPSet = dict(eleEtaBinning = [-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1],
                     eleEtaBinning2D = [-2.1,-1.5,-0.6,0,0.6,1.5,2.1],
                     elePtBinning = [0,25,30,32.5,35,40,45,50,60,80,120,200,400],
                     elePtBinning2D = [0,25,30,40,50,60,80,100,200,400],
                     jetPtBinning = [0,30,40,50,60,80,120,200,400],
                     jetPtBinning2D = [0,30,40,60,80,100,200,400],
                     HTBinning  = [0,100,120,140,150,160,175,200,300,400,500,700],
                     HTBinning2D  = [0,100,125,150.175,200,400,700]),
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*'])
    #denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu24_v*'])
)
###
### SingleMuon
###

topSingleMuonHLTMonitor_Mu24 = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/SingleLepton/SingleMuon/Mu24/',
    enable2DPlots = False,
    nmuons = 1,
    nelectrons = 0,
    njets = 0,
    eleSelection = 'pt>30 & abs(eta)<2.4',
    muoSelection = 'pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15',
    jetSelection = 'pt>20 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu24_eta2p1_v*', 'HLT_IsoMu24_v*'])
)
### ---

topSingleMuonHLTMonitor_Mu27 = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/SingleLepton/SingleMuon/Mu27/',
    enable2DPlots = False,
    nmuons = 1,
    nelectrons = 0,
    njets = 0,
    eleSelection = 'pt>30 & abs(eta)<2.4',
    muoSelection = 'pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15',
    jetSelection = 'pt>20 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*'])
)
### ---

topSingleMuonHLTMonitor_Mu50 = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/SingleLepton/SingleMuon/Mu50/',
    enable2DPlots = False,
    nmuons = 1,
    nelectrons = 0,
    njets = 0,
    eleSelection = 'pt>30 & abs(eta)<2.4',
    muoSelection = 'pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15',
    jetSelection = 'pt>20 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu50_v*'])
)
###
### DiElectron
###

topDiElectronHLTMonitor = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/DiLepton/DiElectron/Ele23Ele12/',
    enable2DPlots = False,
    nmuons = 0,
    nelectrons = 2,
    njets = 0,
    eleSelection = 'pt>15 & abs(eta)<2.4',
    muoSelection = 'pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*'])
)
### ---

topDiElectronHLTMonitor_Dz = topDiElectronHLTMonitor.clone(
    FolderName = 'HLT/TOP/DiLepton/DiElectron/Ele23Ele12_DzEfficiency/',
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v*'])
)
###
### DiMuon
###

topDiMuonHLTMonitor_noDz = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/DiLepton/DiMuon/Mu17_Mu8/',
    enable2DPlots = False,
    nmuons = 2,
    nelectrons = 0,
    njets = 0,
    eleSelection = 'pt>15 & abs(eta)<2.4',
    muoSelection = 'pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*'])
)
### ---

topDiMuonHLTMonitor_Dz = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/DiLepton/DiMuon/Mu17_Mu8_Dz/',
    enable2DPlots = False,
    nmuons = 2,
    nelectrons = 0,
    njets = 0,
    eleSelection = 'pt>15 & abs(eta)<2.4',              
    muoSelection = 'pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    numGenericTriggerEventPSet =dict(hltPaths = ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*'])
)
### ---

topDiMuonHLTMonitor_Dz_Mu17_Mu8 = topDiMuonHLTMonitor_Dz.clone(
    FolderName = 'HLT/TOP/DiLepton/DiMuon/Mu17_Mu8_DzEfficiency/',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*'])
)
### ---

topDiMuonHLTMonitor_Mass8 = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/DiLepton/DiMuon/Mass8/',
    enable2DPlots = False,
    nmuons = 2,
    nelectrons = 0,
    njets = 0,
    eleSelection = 'pt>15 & abs(eta)<2.4',
    muoSelection = 'pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v*'])
)
### ---

topDiMuonHLTMonitor_Mass3p8 = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/DiLepton/DiMuon/Mass3p8/',
    enable2DPlots = False,
    nmuons = 2,
    nelectrons = 0,
    njets = 0,
    eleSelection = 'pt>15 & abs(eta)<2.4',
    muoSelection = 'pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v*'])
)
### ---

topDiMuonHLTMonitor_Mass8Mon = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/DiLepton/DiMuon/Mu17_Mu8_Mass8Efficiency/',
    enable2DPlots = False,
    nmuons = 2,
    nelectrons = 0,
    njets = 0,
    eleSelection = 'pt>15 & abs(eta)<2.4',
    muoSelection = 'pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*'])
)
### ---

topDiMuonHLTMonitor_Mass3p8Mon = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/DiLepton/DiMuon/Mu17_Mu8_Mass3p8Efficiency/',
    enable2DPlots = False,
    nmuons = 2,
    nelectrons = 0,
    njets = 0,
    eleSelection = 'pt>15 & abs(eta)<2.4',
    muoSelection = 'pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*'])
)
###
### ElecMuon
###

topElecMuonHLTMonitor = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/DiLepton/ElecMuon/OR/',
    enable2DPlots = False,
    nmuons = 1,
    nelectrons = 1,
    njets = 0,
    eleSelection = 'pt>15 & abs(eta)<2.4',
    muoSelection = 'pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)',
    jetSelection = 'pt>30 & abs(eta)<2.4',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*',
                                                  'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*',
                                                  'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*',
                                                  'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*',
                                                  'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*',
                                                  'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v*'])
)
### ---

topElecMuonHLTMonitor_Dz_Mu12Ele23 = topElecMuonHLTMonitor.clone(
    FolderName = 'HLT/TOP/DiLepton/ElecMuon/Mu12Ele23_DzEfficiency/',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*'])
)
### ---

topElecMuonHLTMonitor_Dz_Mu8Ele23 = topElecMuonHLTMonitor.clone(
    FolderName = 'HLT/TOP/DiLepton/ElecMuon/Mu8Ele23_DzEfficiency/',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*'])
)
### ---

topElecMuonHLTMonitor_Dz_Mu23Ele12 = topElecMuonHLTMonitor.clone(
    FolderName = 'HLT/TOP/DiLepton/ElecMuon/Mu23Ele12_DzEfficiency/',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v*'])
)
### ---

topElecMuonHLTMonitor_Mu12Ele23 = topElecMuonHLTMonitor.clone(
    FolderName = 'HLT/TOP/DiLepton/ElecMuon/Mu12Ele23/',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*'])
)
### ---

topElecMuonHLTMonitor_Mu8Ele23 = topElecMuonHLTMonitor.clone(
    FolderName = 'HLT/TOP/DiLepton/ElecMuon/Mu8Ele23/',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*'])
)
### ---

topElecMuonHLTMonitor_Mu23Ele12 = topElecMuonHLTMonitor.clone(
    FolderName = 'HLT/TOP/DiLepton/ElecMuon/Mu23Ele12/',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*'])
)
### ---

topElecMuonHLTMonitor_Mu12Ele23_ref = topElecMuonHLTMonitor.clone(
    FolderName = 'HLT/TOP/DiLepton/ElecMuon/Mu12Ele23_Ref/',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*'])
)
### ---

topElecMuonHLTMonitor_Mu8Ele23_ref = topElecMuonHLTMonitor.clone(
    FolderName = 'HLT/TOP/DiLepton/ElecMuon/Mu8Ele23_Ref/',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*'])
)
### ---

topElecMuonHLTMonitor_Mu23Ele12_ref = topElecMuonHLTMonitor.clone(
    FolderName = 'HLT/TOP/DiLepton/ElecMuon/Mu23Ele12_Ref/',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v*'])
)
###
### FullyHadronic
###

fullyhadronic_ref350 = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/FullyHadronic/Reference/PFHT350Monitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 250,
    # Binning
    histoPSet =dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000 ),
                    HTBinning = [0,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900,1000]),
    # Trigger
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT350_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*'])
)
### ---

fullyhadronic_ref370 = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/FullyHadronic/Reference/PFHT370Monitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 250,
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     HTBinning = [0,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900,1000]),
    # Trigger
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT370_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*'])
)
### ---

fullyhadronic_ref430 = hltTOPmonitoring.clone(
    FolderName = 'HLT/TOP/FullyHadronic/Reference/PFHT430Monitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 250,
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000 ),
                     HTBinning = [0,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900,1000]),
    # Trigger
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT430_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*'])
)
### ---

fullyhadronic_DoubleBTag_all = hltTOPmonitoring.clone(
    FolderName   = 'HLT/TOP/FullyHadronic/DoubleBTag/GlobalMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>40 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 2,
    bjetSelection    = 'pt>40 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.4941, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers 
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*'])
)
# conditions 2016
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_all, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_all, workingpoint = 0.8484)
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_all.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_SixJet30_DoubleBTagCSV_p056_v*'])
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_all, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_all, workingpoint = 0.8484)
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_all.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT380_SixPFJet32_DoublePFBTagCSV_2p2_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_all, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_all, workingpoint = 0.4941)
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_all.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94_v*'])
### ---

fullyhadronic_DoubleBTag_DeepJet_all = hltTOPmonitoring.clone(
    FolderName   = 'HLT/TOP/FullyHadronic/DoubleBTagDeepJet/GlobalMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>40 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 2,
    bjetSelection    = 'pt>40 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.4941, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    #btagAlgos        = ["pfDeepFlavourJetTags:probb", "pfDeepFlavourJetTags:probbb", "pfDeepFlavourJetTags:problepb"],
    #workingpoint     = 0.2770, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X )
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers 
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT400_SixPFJet32_DoublePFBTagDeepJet_2p94_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*'])
)

fullyhadronic_DoubleBTag_DeepJet_bjet = hltTOPmonitoring.clone(
    FolderName   = 'HLT/TOP/FullyHadronic/DoubleBTagDeepJet/BJetMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>40 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 2,
    bjetSelection    = 'pt>40 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.1522, # Loose (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    #btagAlgos        = ["pfDeepFlavourJetTags:probb", "pfDeepFlavourJetTags:probbb", "pfDeepFlavourJetTags:problepb"],
    #workingpoint     = 0.0494, # Loose (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X )
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT400_SixPFJet32_DoublePFBTagDeepJet_2p94_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT400_SixPFJet32_v*'])
)
### ---

fullyhadronic_DoubleBTag_jet = hltTOPmonitoring.clone(
    FolderName   = 'HLT/TOP/FullyHadronic/DoubleBTag/JetMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>30 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 2,
    bjetSelection    = 'pt>30 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.4941, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning 
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT400_SixPFJet32_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT370_v*'])
)
# conditions 2016
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_jet, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_jet, workingpoint = 0.8484)
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_jet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_SixJet30_v*'])
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_jet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT350_v*'])
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_jet, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_jet, workingpoint = 0.8484)
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_jet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT380_SixPFJet32_v*'])
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_jet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT370_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_jet, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_jet, workingpoint = 0.4941)
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_jet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_SixPFJet32_v*'])
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_jet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT370_v*'])
### ---

fullyhadronic_DoubleBTag_bjet = hltTOPmonitoring.clone(
    FolderName   = 'HLT/TOP/FullyHadronic/DoubleBTag/BJetMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>40 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 2,
    bjetSelection    = 'pt>40 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.1522, # Loose (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT400_SixPFJet32_v*'])
)
# conditions 2016
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_bjet, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_bjet, workingpoint = 0.5426)
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_bjet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_SixJet30_DoubleBTagCSV_p056_v*'])
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_bjet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_SixJet30_v*'])
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_bjet, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_bjet, workingpoint = 0.5426)
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_bjet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT380_SixPFJet32_DoublePFBTagCSV_2p2_v*'])
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_bjet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT380_SixPFJet32_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_bjet, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_bjet, workingpoint = 0.1522)
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_bjet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94_v*'])
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_bjet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_SixPFJet32_v*'])

### ---

fullyhadronic_DoubleBTag_ref = hltTOPmonitoring.clone(
    FolderName   = 'HLT/TOP/FullyHadronic/DoubleBTag/RefMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>40 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 0,
    bjetSelection    = 'pt>40 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.4941, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,360,380,400,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT400_SixPFJet32_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*'])
)
# conditions 2016
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_ref, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_ref, workingpoint = 0.8484)
run2_HLTconditions_2016.toModify(fullyhadronic_DoubleBTag_ref.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_SixJet30_v*'])
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_ref, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_ref, workingpoint = 0.8484)
run2_HLTconditions_2017.toModify(fullyhadronic_DoubleBTag_ref.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT380_SixPFJet32_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_ref, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_ref, workingpoint = 0.4941)
run2_HLTconditions_2018.toModify(fullyhadronic_DoubleBTag_ref.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_SixPFJet32_v*'])

### ---

fullyhadronic_SingleBTag_all = hltTOPmonitoring.clone(
    FolderName= 'HLT/TOP/FullyHadronic/SingleBTag/GlobalMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>40 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 2,
    bjetSelection    = 'pt>40 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.4941, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*'])
)
# conditions 2016
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_all, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_all, workingpoint = 0.8484)
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_all.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT450_SixJet40_BTagCSV_p056_v*'])
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_all, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_all, workingpoint = 0.8484)
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_all.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT430_SixPFJet40_PFBTagCSV_1p5_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_all, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_all, workingpoint = 0.4941)
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_all.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59_v*'])

### ---

fullyhadronic_SingleBTagDeepJet_all = hltTOPmonitoring.clone(
    FolderName= 'HLT/TOP/FullyHadronic/SingleBTagDeepJet/GlobalMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>40 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 2,
    bjetSelection    = 'pt>40 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.4941, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    #btagAlgos        = ["pfDeepFlavourJetTags:probb", "pfDeepFlavourJetTags:probbb", "pfDeepFlavourJetTags:problepb"],
    #workingpoint     = 0.2770, 
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT450_SixPFJet36_PFBTagDeepJet_1p59_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*']),
)

fullyhadronic_SingleBTagDeepJet_bjet = hltTOPmonitoring.clone(
    FolderName= 'HLT/TOP/FullyHadronic/SingleBTagDeepJet/BJetMonitor/',
    enable2DPlots = False,
    # Selection
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>40 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 2,
    bjetSelection    = 'pt>40 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.1522, # Loose (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    #btagAlgos        = ["pfDeepFlavourJetTags:probb", "pfDeepFlavourJetTags:probbb", "pfDeepFlavourJetTags:problepb"],
    #workingpoint     = 0.0494, # Loose (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X )
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT450_SixPFJet36_PFBTagDeepJet_1p59_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT450_SixPFJet36_v*'])
)

### ---

fullyhadronic_SingleBTag_jet = hltTOPmonitoring.clone(
    FolderName= 'HLT/TOP/FullyHadronic/SingleBTag/JetMonitor/',
    enable2DPlots = False,
    # Selection
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>30 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 2,
    bjetSelection    = 'pt>30 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.4941, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                 jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                 HTBinning    = [0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT450_SixPFJet36_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT430_v*'])
)
# conditions 2016
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_jet, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_jet, workingpoint = 0.8484)
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_jet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT450_SixJet40_v*'])
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_jet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT400_v*'])
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_jet, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_jet, workingpoint = 0.8484)
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_jet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT430_SixPFJet40_v*'])
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_jet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT430_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_jet, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_jet, workingpoint = 0.4941)
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_jet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT450_SixPFJet36_v*'])
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_jet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT430_v*'])

### ---

fullyhadronic_SingleBTag_bjet = hltTOPmonitoring.clone(
    FolderName= 'HLT/TOP/FullyHadronic/SingleBTag/BJetMonitor/',
    enable2DPlots = False,
    # Selection
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>40 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 2,
    bjetSelection    = 'pt>40 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.1522, # Loose (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT450_SixPFJet36_v*'])
)

# conditions 2016
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_bjet, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_bjet, workingpoint = 0.5426)
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_bjet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT450_SixJet40_BTagCSV_p056_v*'])
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_bjet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT450_SixJet40_v*'])
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_bjet, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_bjet, workingpoint = 0.5426)
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_bjet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT430_SixPFJet40_PFBTagCSV_1p5_v*'])
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_bjet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT430_SixPFJet40_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_bjet, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_bjet, workingpoint = 0.1522)
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_bjet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59_v*'])
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_bjet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT450_SixPFJet36_v*'])

### ---

fullyhadronic_SingleBTag_ref = hltTOPmonitoring.clone(
    FolderName= 'HLT/TOP/FullyHadronic/SingleBTag/RefMonitor/',
    enable2DPlots = False,
    # Selection
    leptJetDeltaRmin = 0.0,
    njets            = 6,
    jetSelection     = 'pt>40 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 0,
    bjetSelection    = 'pt>40 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.4941, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                     jetPtBinning = [0,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200],
                     HTBinning    = [0,400,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT450_SixPFJet36_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*'])
)
# conditions 2016
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_ref, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_ref, workingpoint = 0.8484)
run2_HLTconditions_2016.toModify(fullyhadronic_SingleBTag_ref.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT450_SixJet40_v*'])
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_ref, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_ref, workingpoint = 0.8484)
run2_HLTconditions_2017.toModify(fullyhadronic_SingleBTag_ref.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT430_SixPFJet40_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_ref, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_ref, workingpoint = 0.4941)
run2_HLTconditions_2018.toModify(fullyhadronic_SingleBTag_ref.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT450_SixPFJet36_v*'])

### ---

fullyhadronic_TripleBTag_all = hltTOPmonitoring.clone(
    FolderName   = 'HLT/TOP/FullyHadronic/TripleBTag/GlobalMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 4,
    jetSelection     = 'pt>45 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 4,
    bjetSelection    = 'pt>45 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.4941, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning
    histoPSet = dict(htPSet =dict(nbins= 50, xmin= 0.0, xmax= 1000),
                 jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400],
                 HTBinning    = [0,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_IsoMu27_v*'])
)
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_TripleBTag_all, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_TripleBTag_all, workingpoint = 0.8484)
run2_HLTconditions_2017.toModify(fullyhadronic_TripleBTag_all.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_TripleBTag_all, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_TripleBTag_all, workingpoint = 0.4941)
run2_HLTconditions_2018.toModify(fullyhadronic_TripleBTag_all.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5_v*'])
### ---

fullyhadronic_TripleBTag_jet = hltTOPmonitoring.clone(
    FolderName   = 'HLT/TOP/FullyHadronic/TripleBTag/JetMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 4,
    jetSelection     = 'pt>45 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 4,
    bjetSelection    = 'pt>45 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.4941, # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                 jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400],
                 HTBinning    = [0,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT350_v*'])
)
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_TripleBTag_jet, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_TripleBTag_jet, workingpoint = 0.8484)
run2_HLTconditions_2017.toModify(fullyhadronic_TripleBTag_jet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT300PT30_QuadPFJet_75_60_45_40_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_TripleBTag_jet, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_TripleBTag_jet, workingpoint = 0.4941)
run2_HLTconditions_2018.toModify(fullyhadronic_TripleBTag_jet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v*'])

### ---

fullyhadronic_TripleBTag_bjet = hltTOPmonitoring.clone(
    FolderName   = 'HLT/TOP/FullyHadronic/TripleBTag/BJetMonitor/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 4,
    jetSelection     = 'pt>45 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 4,
    bjetSelection    = 'pt>45 & abs(eta)<2.4',
    btagAlgos        = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"],
    workingpoint     = 0.1522, # Loose (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                 jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400],
                 HTBinning    = [0,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers 
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v*'])
)
# conditions 2017
run2_HLTconditions_2017.toModify(fullyhadronic_TripleBTag_bjet, btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"])
run2_HLTconditions_2017.toModify(fullyhadronic_TripleBTag_bjet, workingpoint = 0.5426)
run2_HLTconditions_2017.toModify(fullyhadronic_TripleBTag_bjet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0_v*'])
run2_HLTconditions_2017.toModify(fullyhadronic_TripleBTag_bjet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT300PT30_QuadPFJet_75_60_45_40_v*'])
# conditions 2018
run2_HLTconditions_2018.toModify(fullyhadronic_TripleBTag_bjet, btagAlgos = ["pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"])
run2_HLTconditions_2018.toModify(fullyhadronic_TripleBTag_bjet, workingpoint = 0.1522)
run2_HLTconditions_2018.toModify(fullyhadronic_TripleBTag_bjet.numGenericTriggerEventPSet, hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5_v*'])
run2_HLTconditions_2018.toModify(fullyhadronic_TripleBTag_bjet.denGenericTriggerEventPSet, hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v*'])

###
### Top HLT-DQM Sequence
###

from DQMOffline.Trigger.HLTEGTnPMonitor_cfi import egmGsfElectronIDsForDQM

topMonitorHLT = cms.Sequence(

      topEleJet_ele
    + topEleJet_jet
    + topEleJet_all
    + topEleHT_ele
    + topEleHT_ht
    + topEleHT_all

    + topSingleMuonHLTMonitor_Mu24
    + topSingleMuonHLTMonitor_Mu27
    + topSingleMuonHLTMonitor_Mu50

    + topDiElectronHLTMonitor
    + topDiElectronHLTMonitor_Dz

    + topDiMuonHLTMonitor_noDz
    + topDiMuonHLTMonitor_Dz
    + topDiMuonHLTMonitor_Dz_Mu17_Mu8
    + topDiMuonHLTMonitor_Mass8
    + topDiMuonHLTMonitor_Mass3p8
    + topDiMuonHLTMonitor_Mass8Mon
    + topDiMuonHLTMonitor_Mass3p8Mon

    + topElecMuonHLTMonitor
    + topElecMuonHLTMonitor_Dz_Mu12Ele23
    + topElecMuonHLTMonitor_Dz_Mu8Ele23
    + topElecMuonHLTMonitor_Dz_Mu23Ele12
    + topElecMuonHLTMonitor_Mu12Ele23
    + topElecMuonHLTMonitor_Mu8Ele23
    + topElecMuonHLTMonitor_Mu23Ele12
    + topElecMuonHLTMonitor_Mu12Ele23_ref
    + topElecMuonHLTMonitor_Mu8Ele23_ref
    + topElecMuonHLTMonitor_Mu23Ele12_ref

    + fullyhadronic_ref350
    + fullyhadronic_ref370
    + fullyhadronic_ref430

    + fullyhadronic_DoubleBTag_all  
    + fullyhadronic_DoubleBTag_jet
    + fullyhadronic_DoubleBTag_bjet
    + fullyhadronic_DoubleBTag_ref

    + fullyhadronic_DoubleBTag_DeepJet_all
    + fullyhadronic_DoubleBTag_DeepJet_bjet

    + fullyhadronic_SingleBTag_all
    + fullyhadronic_SingleBTag_jet
    + fullyhadronic_SingleBTag_bjet
    + fullyhadronic_SingleBTag_ref

    + fullyhadronic_SingleBTagDeepJet_all
    + fullyhadronic_SingleBTagDeepJet_bjet

    + fullyhadronic_TripleBTag_all
    + fullyhadronic_TripleBTag_jet
    + fullyhadronic_TripleBTag_bjet
    , cms.Task(egmGsfElectronIDsForDQM) # Use of electron VID requires this module being executed first
)

topHLTDQMSourceExtra = cms.Sequence(
)

topMonitorHLT_2016 = topMonitorHLT.copy()
topMonitorHLT_2016.remove( topEleJet_jet )
topMonitorHLT_2016.remove( topEleJet_ele )
topMonitorHLT_2016.remove( topEleJet_all )
topMonitorHLT_2016.remove( topEleHT_ht )
topMonitorHLT_2016.remove( topEleHT_ele )
topMonitorHLT_2016.remove( topEleHT_all )
topMonitorHLT_2016.remove( topDiMuonHLTMonitor_Mass3p8 )
topMonitorHLT_2016.remove( topDiMuonHLTMonitor_Mass8Mon )
topMonitorHLT_2016.remove( topDiMuonHLTMonitor_Mass3p8Mon )
topMonitorHLT_2016.remove( fullyhadronic_ref370 )
topMonitorHLT_2016.remove( fullyhadronic_ref430 )
topMonitorHLT_2016.remove( fullyhadronic_TripleBTag_all )
topMonitorHLT_2016.remove( fullyhadronic_TripleBTag_jet )
topMonitorHLT_2016.remove( fullyhadronic_TripleBTag_bjet )

run2_HLTconditions_2016.toReplaceWith(topMonitorHLT, topMonitorHLT_2016)
