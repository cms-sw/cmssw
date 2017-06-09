import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

eleJet_jet = hltTOPmonitoring.clone()
eleJet_jet.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleJet/JetMonitor')
eleJet_jet.nmuons = cms.uint32(0)
eleJet_jet.nelectrons = cms.uint32(1)
eleJet_jet.njets = cms.uint32(1)
eleJet_jet.eleSelection = cms.string('pt>50 & abs(eta)<2.1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
eleJet_jet.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
eleJet_jet.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1)
eleJet_jet.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.7,-1.2,-0.6,-0.3,0,0.3,0.6,1.2,1.7,2.1)
eleJet_jet.histoPSet.elePtBinning = cms.vdouble(0,50,60,80,120,200,400)
eleJet_jet.histoPSet.elePtBinning2D = cms.vdouble(0,50,70,120,200,400)
eleJet_jet.histoPSet.jetPtBinning = cms.vdouble(0,30,32.5,35,37.5,40,45,50,60,80,120,200,400)
eleJet_jet.histoPSet.jetPtBinning2D = cms.vdouble(0,30,35,40,45,50,60,80,100,200,400)
eleJet_jet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*')
eleJet_jet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_v*',
                                                             'HLT_Ele35_WPTight_Gsf_v*',
                                                             'HLT_Ele38_WPTight_Gsf_v*',
                                                             'HLT_Ele40_WPTight_Gsf_v*',)

eleJet_ele = hltTOPmonitoring.clone()
eleJet_ele.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleJet/ElectronMonitor')
eleJet_ele.nmuons = cms.uint32(0)
eleJet_ele.nelectrons = cms.uint32(1)
eleJet_ele.njets = cms.uint32(1)
eleJet_ele.eleSelection = cms.string('pt>25 & abs(eta)<2.1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
eleJet_ele.jetSelection = cms.string('pt>50 & abs(eta)<2.4')
eleJet_ele.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1)
eleJet_ele.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.7,-1.2,-0.6,-0.3,0,0.3,0.6,1.2,1.7,2.1)
eleJet_ele.histoPSet.elePtBinning = cms.vdouble(0,25,27.5,30,32.5,35,40,45,50,60,80,120,200,400)
eleJet_ele.histoPSet.elePtBinning2D = cms.vdouble(0,25,27.5,30,35,40,50,60,80,100,200,400)
eleJet_ele.histoPSet.jetPtBinning = cms.vdouble(0,50,60,80,120,200,400)
eleJet_ele.histoPSet.jetPtBinning2D = cms.vdouble(0,50,60,80,100,200,400)
eleJet_ele.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*')
eleJet_ele.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet60_v*')

eleJet_all = hltTOPmonitoring.clone()
eleJet_all.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleJet/GlobalMonitor')
eleJet_all.nmuons = cms.uint32(0)
eleJet_all.nelectrons = cms.uint32(1)
eleJet_all.njets = cms.uint32(1)
eleJet_all.eleSelection = cms.string('pt>25 & abs(eta)<2.1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
eleJet_all.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
eleJet_all.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1)
eleJet_all.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.7,-1.2,-0.6,-0.3,0,0.3,0.6,1.2,1.7,2.1)
eleJet_all.histoPSet.elePtBinning = cms.vdouble(0,25,27.5,30,32.5,35,40,45,50,60,80,120,200,400)
eleJet_all.histoPSet.elePtBinning2D = cms.vdouble(0,25,27.5,30,35,40,50,60,80,100,200,400)
eleJet_all.histoPSet.jetPtBinning = cms.vdouble(0,30,32.5,35,37.5,40,50,60,80,120,200,400)
eleJet_all.histoPSet.jetPtBinning2D = cms.vdouble(0,30,35,40,45,50,60,80,100,200,400)
eleJet_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*')
# eleJet_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu24_v*')


eleHT_ht = hltTOPmonitoring.clone()
eleHT_ht.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleHT/HTMonitor')
eleHT_ht.nmuons = cms.uint32(0)
eleHT_ht.nelectrons = cms.uint32(1)
eleHT_ht.njets = cms.uint32(2)
eleHT_ht.eleSelection = cms.string('pt>50 & abs(eta)<2.1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
eleHT_ht.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
eleHT_ht.HTcut = cms.double(100)
eleHT_ht.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1)
eleHT_ht.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.7,-1.2,-0.6,-0.3,0,0.3,0.6,1.2,1.7,2.1)
eleHT_ht.histoPSet.elePtBinning = cms.vdouble(0,50,60,80,120,200,400)
eleHT_ht.histoPSet.elePtBinning2D = cms.vdouble(0,50,70,120,200,400)
eleHT_ht.histoPSet.jetPtBinning = cms.vdouble(0,30,40,50,60,80,120,200,400)
eleHT_ht.histoPSet.jetPtBinning2D = cms.vdouble(0,30,40,60,80,100,200,400)
eleHT_ht.histoPSet.HTBinning  = cms.vdouble(0,100,120,140,150,160,175,200,300,400,500,700)
eleHT_ht.histoPSet.HTBinning2D  = cms.vdouble(0,100,125,150.175,200,400,700)
eleHT_ht.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*')
eleHT_ht.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_v*',
                                                           'HLT_Ele35_WPTight_Gsf_v*',
                                                           'HLT_Ele38_WPTight_Gsf_v*',
                                                           'HLT_Ele40_WPTight_Gsf_v*',)

eleHT_ele = hltTOPmonitoring.clone()
eleHT_ele.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleHT/ElectronMonitor')
eleHT_ele.nmuons = cms.uint32(0)
eleHT_ele.nelectrons = cms.uint32(1)
eleHT_ele.njets = cms.uint32(2)
eleHT_ele.eleSelection = cms.string('pt>25 & abs(eta)<2.1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
eleHT_ele.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
eleHT_ele.HTcut = cms.double(200)
eleHT_ele.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1)
eleHT_ele.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.7,-1.2,-0.6,-0.3,0,0.3,0.6,1.2,1.7,2.1)
eleHT_ele.histoPSet.elePtBinning = cms.vdouble(0,25,27.5,30,32.5,35,40,45,50,60,80,120,200,400)
eleHT_ele.histoPSet.elePtBinning2D = cms.vdouble(0,25,27.5,30,35,40,50,60,80,100,200,400)
eleHT_ele.histoPSet.jetPtBinning = cms.vdouble(0,30,40,50,60,80,120,200,400)
eleHT_ele.histoPSet.jetPtBinning2D = cms.vdouble(0,30,40,60,80,100,200,400)
eleHT_ele.histoPSet.HTBinning  = cms.vdouble(0,200,250,300,350,400,500,700)
eleHT_ele.histoPSet.HTBinning2D  = cms.vdouble(0,200,250,300,400,500,700)
eleHT_ele.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*')
eleHT_ele.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_HT200_v*',
                                                            'HLT_HT275_v*',)

eleHT_all = hltTOPmonitoring.clone()
eleHT_all.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleHT/GlobalMonitor')
eleHT_all.nmuons = cms.uint32(0)
eleHT_all.nelectrons = cms.uint32(1)
eleHT_all.njets = cms.uint32(2)
eleHT_all.eleSelection = cms.string('pt>25 & abs(eta)<2.1 & (dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.1')
eleHT_all.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
eleHT_all.HTcut = cms.double(100)
eleHT_all.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1)
eleHT_all.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.7,-1.2,-0.6,-0.3,0,0.3,0.6,1.2,1.7,2.1)
eleHT_all.histoPSet.elePtBinning = cms.vdouble(0,25,27.5,30,32.5,35,40,45,50,60,80,120,200,400)
eleHT_all.histoPSet.elePtBinning2D = cms.vdouble(0,25,27.5,30,35,40,50,60,80,100,200,400)
eleHT_all.histoPSet.jetPtBinning = cms.vdouble(0,30,40,50,60,80,120,200,400)
eleHT_all.histoPSet.jetPtBinning2D = cms.vdouble(0,30,40,60,80,100,200,400)
eleHT_all.histoPSet.HTBinning  = cms.vdouble(0,100,120,140,150,160,175,200,300,400,500,700)
eleHT_all.histoPSet.HTBinning2D  = cms.vdouble(0,100,125,150.175,200,400,700)
eleHT_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*')
# eleHT_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu24_v*')


#ATHER
#########SingleMuon
topSingleMuonHLTMonitor_Mu20 = hltTOPmonitoring.clone()
topSingleMuonHLTMonitor_Mu20.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/SingleLepton/SingleMuon/Mu20/')
topSingleMuonHLTMonitor_Mu20.nmuons = cms.uint32(1)
topSingleMuonHLTMonitor_Mu20.nelectrons = cms.uint32(0)
topSingleMuonHLTMonitor_Mu20.njets = cms.uint32(4)
topSingleMuonHLTMonitor_Mu20.eleSelection = cms.string('pt>30 & abs(eta)<2.4 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')
topSingleMuonHLTMonitor_Mu20.muoSelection = cms.string('pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15')
topSingleMuonHLTMonitor_Mu20.jetSelection = cms.string('pt>20 & abs(eta)<2.4')
topSingleMuonHLTMonitor_Mu20.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Mu20_v*', 'HLT_TkMu20_v*' , 'HLT_IsoMu20_v*',  'HLT_IsoTkMu20_v*'])


topSingleMuonHLTMonitor_Mu24 = hltTOPmonitoring.clone()
topSingleMuonHLTMonitor_Mu24.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/SingleLepton/SingleMuon/Mu24/')
topSingleMuonHLTMonitor_Mu24.nmuons = cms.uint32(1)
topSingleMuonHLTMonitor_Mu24.nelectrons = cms.uint32(0)
topSingleMuonHLTMonitor_Mu24.njets = cms.uint32(4)
topSingleMuonHLTMonitor_Mu24.eleSelection = cms.string('pt>30 & abs(eta)<2.4 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')
topSingleMuonHLTMonitor_Mu24.muoSelection = cms.string('pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15')
topSingleMuonHLTMonitor_Mu24.jetSelection = cms.string('pt>20 & abs(eta)<2.4')
topSingleMuonHLTMonitor_Mu24.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_IsoMu24_eta2p1_v*', 'HLT_IsoMu24_v*', 'HLT_IsoTkMu24_eta2p1_v*', 'HLT_IsoTkMu24_v*'])

topSingleMuonHLTMonitor_Mu27 = hltTOPmonitoring.clone()
topSingleMuonHLTMonitor_Mu27.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/SingleLepton/SingleMuon/Mu27/')
topSingleMuonHLTMonitor_Mu27.nmuons = cms.uint32(1)
topSingleMuonHLTMonitor_Mu27.nelectrons = cms.uint32(0)
topSingleMuonHLTMonitor_Mu27.njets = cms.uint32(4)
topSingleMuonHLTMonitor_Mu27.eleSelection = cms.string('pt>30 & abs(eta)<2.4 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')
topSingleMuonHLTMonitor_Mu27.muoSelection = cms.string('pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15')
topSingleMuonHLTMonitor_Mu27.jetSelection = cms.string('pt>20 & abs(eta)<2.4')
topSingleMuonHLTMonitor_Mu27.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Mu27_v*', 'HLT_TkMu27_v*', 'HLT_IsoMu27_v*', 'HLT_IsoTkMu27_v*'])


topSingleMuonHLTMonitor_Mu50 = hltTOPmonitoring.clone()
topSingleMuonHLTMonitor_Mu50.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/SingleLepton/SingleMuon/Mu50/')
topSingleMuonHLTMonitor_Mu50.nmuons = cms.uint32(1)
topSingleMuonHLTMonitor_Mu50.nelectrons = cms.uint32(0)
topSingleMuonHLTMonitor_Mu50.njets = cms.uint32(4)
topSingleMuonHLTMonitor_Mu50.eleSelection = cms.string('pt>30 & abs(eta)<2.4 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')
topSingleMuonHLTMonitor_Mu50.muoSelection = cms.string('pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15')
topSingleMuonHLTMonitor_Mu50.jetSelection = cms.string('pt>20 & abs(eta)<2.4')
topSingleMuonHLTMonitor_Mu50.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_TkMu50_v*', 'HLT_Mu50_v*'])


#########DiElectron
topDiElectronHLTMonitor = hltTOPmonitoring.clone()
topDiElectronHLTMonitor.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/DiLepton/DiElectron/')
topDiElectronHLTMonitor.nmuons = cms.uint32(0)
topDiElectronHLTMonitor.nelectrons = cms.uint32(2)
topDiElectronHLTMonitor.njets = cms.uint32(2)
topDiElectronHLTMonitor.eleSelection = cms.string('pt>20 & abs(eta)<2.4  & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')

topDiElectronHLTMonitor.muoSelection = cms.string('pt>20 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')

topDiElectronHLTMonitor.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiElectronHLTMonitor.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*'])


#########DiMuon
topDiMuonHLTMonitor_noDz = hltTOPmonitoring.clone()
topDiMuonHLTMonitor_noDz.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/DiLepton/DiMuon/NoDz/')
topDiMuonHLTMonitor_noDz.nmuons = cms.uint32(2)
topDiMuonHLTMonitor_noDz.nelectrons = cms.uint32(0)
topDiMuonHLTMonitor_noDz.njets = cms.uint32(2)
topDiMuonHLTMonitor_noDz.eleSelection = cms.string('pt>20 & abs(eta)<2.4  & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')

topDiMuonHLTMonitor_noDz.muoSelection = cms.string('pt>20 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')

topDiMuonHLTMonitor_noDz.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiMuonHLTMonitor_noDz.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*','HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*'])

topDiMuonHLTMonitor_Dz = hltTOPmonitoring.clone()
topDiMuonHLTMonitor_Dz.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/DiLepton/DiMuon/Dz/')
topDiMuonHLTMonitor_Dz.nmuons = cms.uint32(2)
topDiMuonHLTMonitor_Dz.nelectrons = cms.uint32(0)
topDiMuonHLTMonitor_Dz.njets = cms.uint32(2)
topDiMuonHLTMonitor_Dz.eleSelection = cms.string('pt>20 & abs(eta)<2.4  & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')              

topDiMuonHLTMonitor_Dz.muoSelection = cms.string('pt>20 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')

topDiMuonHLTMonitor_Dz.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiMuonHLTMonitor_Dz.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*','HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*'])

topDiMuonHLTMonitor_Mass8 = hltTOPmonitoring.clone()
topDiMuonHLTMonitor_Mass8.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/DiLepton/DiMuon/Mass8/')
topDiMuonHLTMonitor_Mass8.nmuons = cms.uint32(2)
topDiMuonHLTMonitor_Mass8.nelectrons = cms.uint32(0)
topDiMuonHLTMonitor_Mass8.njets = cms.uint32(2)
topDiMuonHLTMonitor_Mass8.eleSelection = cms.string('pt>20 & abs(eta)<2.4  & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')

topDiMuonHLTMonitor_Mass8.muoSelection = cms.string('pt>20 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')

topDiMuonHLTMonitor_Mass8.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiMuonHLTMonitor_Mass8.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8'])

#########ElecMuon
topElecMuonHLTMonitor = hltTOPmonitoring.clone()
topElecMuonHLTMonitor.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/DiLepton/ElecMuon/')
topElecMuonHLTMonitor.nmuons = cms.uint32(1)
topElecMuonHLTMonitor.nelectrons = cms.uint32(1)
topElecMuonHLTMonitor.njets = cms.uint32(2)
topElecMuonHLTMonitor.eleSelection = cms.string('pt>20 & abs(eta)<2.4 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')

topElecMuonHLTMonitor.muoSelection = cms.string('pt>20 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)') 

topElecMuonHLTMonitor.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topElecMuonHLTMonitor.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*','HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*'])

# Marina

fullyhadronic_ref350 = hltTOPmonitoring.clone()
fullyhadronic_ref350.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/FullyHadronic/Reference/PFHT350Monitor/')
# Selections
fullyhadronic_ref350.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_ref350.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_ref350.HTcut            = cms.double(250)
# Binning
fullyhadronic_ref350.histoPSet.htPSet = cms.PSet(nbins=cms.int32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_ref350.histoPSet.HTBinning = cms.vdouble(0,240,260,280,300,320,340,360,380,400,420,440,460,480,
                                                       500,520,540,560,580,600,650,700,750,800,850,900,1000)
# Trigger
fullyhadronic_ref350.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT350_v')
fullyhadronic_ref350.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v')


fullyhadronic_ref370 = hltTOPmonitoring.clone()
fullyhadronic_ref370.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/FullyHadronic/Reference/PFHT370Monitor/')
# Selections
fullyhadronic_ref370.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_ref370.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_ref370.HTcut            = cms.double(250)
# Binning
fullyhadronic_ref370.histoPSet.htPSet = cms.PSet(nbins=cms.int32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_ref370.histoPSet.HTBinning = cms.vdouble(0,240,260,280,300,320,340,360,380,400,420,440,460,480,
                                                       500,520,540,560,580,600,650,700,750,800,850,900,1000)
# Trigger
fullyhadronic_ref370.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT370_v')
fullyhadronic_ref370.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v')

fullyhadronic_ref430 = hltTOPmonitoring.clone()
fullyhadronic_ref430.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/FullyHadronic/Reference/PFHT430Monitor/')
# Selections
fullyhadronic_ref430.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_ref430.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_ref430.HTcut            = cms.double(250)
# Binning
fullyhadronic_ref430.histoPSet.htPSet = cms.PSet(nbins=cms.int32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_ref430.histoPSet.HTBinning = cms.vdouble(0,240,260,280,300,320,340,360,380,400,420,440,460,480,
                                                       500,520,540,560,580,600,650,700,750,800,850,900,1000)
# Trigger
fullyhadronic_ref430.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT430_v')
fullyhadronic_ref430.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v')


fullyhadronic_DoubleBTag_all = hltTOPmonitoring.clone()
fullyhadronic_DoubleBTag_all.FolderName   = cms.string('HLT/TopHLTOffline/TopMonitor/FullyHadronic/DoubleBTag/GlobalMonitor/')
# Selections
fullyhadronic_DoubleBTag_all.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_DoubleBTag_all.njets            = cms.uint32(6)
fullyhadronic_DoubleBTag_all.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_all.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_all.HTcut            = cms.double(450)
fullyhadronic_DoubleBTag_all.nbjets           = cms.uint32(2)
fullyhadronic_DoubleBTag_all.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_all.workingpoint     = cms.double(0.8484) # Medium
# Binning
fullyhadronic_DoubleBTag_all.histoPSet.htPSet = cms.PSet(nbins=cms.int32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_DoubleBTag_all.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
fullyhadronic_DoubleBTag_all.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers 
fullyhadronic_DoubleBTag_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT380_SixJet32_DoubleBTagCSV_p075_v')
fullyhadronic_DoubleBTag_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v')

fullyhadronic_DoubleBTag_jet = hltTOPmonitoring.clone()
fullyhadronic_DoubleBTag_jet.FolderName   = cms.string('HLT/TopHLTOffline/TopMonitor/FullyHadronic/DoubleBTag/JetMonitor/')
# Selections
fullyhadronic_DoubleBTag_jet.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_DoubleBTag_jet.njets            = cms.uint32(6)
fullyhadronic_DoubleBTag_jet.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_jet.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_jet.HTcut            = cms.double(450)
fullyhadronic_DoubleBTag_jet.nbjets           = cms.uint32(2)
fullyhadronic_DoubleBTag_jet.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_jet.workingpoint = cms.double(0.8484) # Medium
# Binning 
fullyhadronic_DoubleBTag_jet.histoPSet.htPSet = cms.PSet(nbins=cms.int32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_DoubleBTag_jet.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
fullyhadronic_DoubleBTag_jet.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_DoubleBTag_jet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT380_SixJet32_DoubleBTagCSV_p075_v')
fullyhadronic_DoubleBTag_jet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT370_v')

fullyhadronic_DoubleBTag_bjet = hltTOPmonitoring.clone()
fullyhadronic_DoubleBTag_bjet.FolderName   = cms.string('HLT/TopHLTOffline/TopMonitor/FullyHadronic/DoubleBTag/BJetMonitor/')
# Selections
fullyhadronic_DoubleBTag_bjet.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_DoubleBTag_bjet.njets            = cms.uint32(6)
fullyhadronic_DoubleBTag_bjet.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_bjet.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_bjet.HTcut            = cms.double(450)
fullyhadronic_DoubleBTag_bjet.nbjets           = cms.uint32(2)
fullyhadronic_DoubleBTag_bjet.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_bjet.workingpoint     = cms.double(0.70)
# Binning
fullyhadronic_DoubleBTag_bjet.histoPSet.htPSet = cms.PSet(nbins=cms.int32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_DoubleBTag_bjet.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
fullyhadronic_DoubleBTag_bjet.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_DoubleBTag_bjet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT380_SixJet32_DoubleBTagCSV_p075_v')
fullyhadronic_DoubleBTag_bjet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT380_SixJet32_v')


fullyhadronic_SingleBTag_all = hltTOPmonitoring.clone()
fullyhadronic_SingleBTag_all.FolderName= cms.string('HLT/TopHLTOffline/TopMonitor/FullyHadronic/SingleBTag/GlobalMonitor/')
# Selections
fullyhadronic_SingleBTag_all.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_SingleBTag_all.njets            = cms.uint32(6)
fullyhadronic_SingleBTag_all.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_all.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_all.HTcut            = cms.double(450)
fullyhadronic_SingleBTag_all.nbjets           = cms.uint32(2)
fullyhadronic_SingleBTag_all.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_all.workingpoint     = cms.double(0.8484) # Medium
# Binning
fullyhadronic_SingleBTag_all.histoPSet.htPSet = cms.PSet(nbins=cms.int32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_SingleBTag_all.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
fullyhadronic_SingleBTag_all.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_SingleBTag_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT430_SixJet40_BTagCSV_p080_v')
fullyhadronic_SingleBTag_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v')

fullyhadronic_SingleBTag_jet = hltTOPmonitoring.clone()
fullyhadronic_SingleBTag_jet.FolderName= cms.string('HLT/TopHLTOffline/TopMonitor/FullyHadronic/SingleBTag/JetMonitor/')
# Selection
fullyhadronic_SingleBTag_jet.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_SingleBTag_jet.njets            = cms.uint32(6)
fullyhadronic_SingleBTag_jet.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_jet.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_jet.HTcut            = cms.double(450)
fullyhadronic_SingleBTag_jet.nbjets           = cms.uint32(2)
fullyhadronic_SingleBTag_jet.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_jet.workingpoint     = cms.double(0.8484) # Medium
# Binning
fullyhadronic_SingleBTag_jet.histoPSet.htPSet = cms.PSet(nbins=cms.int32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_SingleBTag_jet.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
fullyhadronic_SingleBTag_jet.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_SingleBTag_jet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT430_SixJet40_BTagCSV_p080_v')
fullyhadronic_SingleBTag_jet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT430_v')

fullyhadronic_SingleBTag_bjet = hltTOPmonitoring.clone()
fullyhadronic_SingleBTag_bjet.FolderName= cms.string('HLT/TopHLTOffline/TopMonitor/FullyHadronic/SingleBTag/BJetMonitor/')
# Selection
fullyhadronic_SingleBTag_bjet.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_SingleBTag_bjet.njets            = cms.uint32(6)
fullyhadronic_SingleBTag_bjet.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_bjet.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_bjet.HTcut            = cms.double(450)
fullyhadronic_SingleBTag_bjet.nbjets           = cms.uint32(2)
fullyhadronic_SingleBTag_bjet.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_bjet.workingpoint     = cms.double(0.70)
# Binning
fullyhadronic_SingleBTag_bjet.histoPSet.htPSet = cms.PSet(nbins=cms.int32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_SingleBTag_bjet.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
fullyhadronic_SingleBTag_bjet.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_SingleBTag_bjet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT430_SixJet40_BTagCSV_p080_v')
fullyhadronic_SingleBTag_bjet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT430_SixJet40_v')



topMonitorHLT = cms.Sequence(
    eleJet_ele
    + eleJet_jet
    + eleJet_all
    + eleHT_ele
    + eleHT_ht
    + eleHT_all
    + topSingleMuonHLTMonitor_Mu20
    + topSingleMuonHLTMonitor_Mu24
    + topSingleMuonHLTMonitor_Mu27
    + topSingleMuonHLTMonitor_Mu50
    + topDiElectronHLTMonitor
    + topDiMuonHLTMonitor_noDz
    + topDiMuonHLTMonitor_Dz
    + topDiMuonHLTMonitor_Mass8
    + topElecMuonHLTMonitor
    + fullyhadronic_ref350
    + fullyhadronic_ref370
    + fullyhadronic_ref430
    + fullyhadronic_DoubleBTag_all
    + fullyhadronic_DoubleBTag_jet
    + fullyhadronic_DoubleBTag_bjet
    + fullyhadronic_SingleBTag_all
    + fullyhadronic_SingleBTag_jet
    + fullyhadronic_SingleBTag_bjet

    )
