import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

topEleJet_jet = hltTOPmonitoring.clone()
topEleJet_jet.FolderName = cms.string('HLT/TOP/EleJet/JetMonitor')
topEleJet_jet.nmuons = cms.uint32(0)
topEleJet_jet.nelectrons = cms.uint32(1)
topEleJet_jet.njets = cms.uint32(1)
topEleJet_jet.eleSelection = cms.string('pt>50 & abs(eta)<2.1')
topEleJet_jet.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topEleJet_jet.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1)
topEleJet_jet.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.5,-0.6,0,0.6,1.5,2.1)
topEleJet_jet.histoPSet.elePtBinning = cms.vdouble(0,50,60,80,120,200,400)
topEleJet_jet.histoPSet.elePtBinning2D = cms.vdouble(0,50,70,120,200,400)
topEleJet_jet.histoPSet.jetPtBinning = cms.vdouble(0,30,35,37.5,40,50,60,80,120,200,400)
topEleJet_jet.histoPSet.jetPtBinning2D = cms.vdouble(0,30,35,40,50,60,80,100,200,400)
topEleJet_jet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*')
topEleJet_jet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele35_WPTight_Gsf_v*',
                                                             'HLT_Ele38_WPTight_Gsf_v*',
                                                             'HLT_Ele40_WPTight_Gsf_v*',)

topEleJet_ele = hltTOPmonitoring.clone()
topEleJet_ele.FolderName = cms.string('HLT/TOP/EleJet/ElectronMonitor')
topEleJet_ele.nmuons = cms.uint32(0)
topEleJet_ele.nelectrons = cms.uint32(1)
topEleJet_ele.njets = cms.uint32(1)
topEleJet_ele.eleSelection = cms.string('pt>25 & abs(eta)<2.1')
topEleJet_ele.jetSelection = cms.string('pt>50 & abs(eta)<2.4')
topEleJet_ele.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1)
topEleJet_ele.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.5,-0.6,0,0.6,1.5,2.1)
topEleJet_ele.histoPSet.elePtBinning = cms.vdouble(0,25,30,32.5,35,40,45,50,60,80,120,200,400)
topEleJet_ele.histoPSet.elePtBinning2D = cms.vdouble(0,25,30,40,50,60,80,100,200,400)
topEleJet_ele.histoPSet.jetPtBinning = cms.vdouble(0,50,60,80,120,200,400)
topEleJet_ele.histoPSet.jetPtBinning2D = cms.vdouble(0,50,60,80,100,200,400)
topEleJet_ele.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*')
topEleJet_ele.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet60_v*',
                                                                'HLT_PFJet80_v*',
                                                                'HLT_PFJet140_v*',
                                                                'HLT_PFJet200_v*',
                                                                'HLT_PFJet260_v*',
                                                                'HLT_PFJet320_v*',
                                                                'HLT_PFJet400_v*',
                                                                'HLT_PFJet450_v*',
                                                                'HLT_PFJet500_v*',
                                                                'HLT_PFJet550_v*',)

topEleJet_all = hltTOPmonitoring.clone()
topEleJet_all.FolderName = cms.string('HLT/TOP/EleJet/GlobalMonitor')
topEleJet_all.nmuons = cms.uint32(0)
topEleJet_all.nelectrons = cms.uint32(1)
topEleJet_all.njets = cms.uint32(1)
topEleJet_all.eleSelection = cms.string('pt>25 & abs(eta)<2.1')
topEleJet_all.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topEleJet_all.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1)
topEleJet_all.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.5,-0.6,0,0.6,1.5,2.1)
topEleJet_all.histoPSet.elePtBinning = cms.vdouble(0,25,30,32.5,35,40,45,50,60,80,120,200,400)
topEleJet_all.histoPSet.elePtBinning2D = cms.vdouble(0,25,30,40,50,60,80,100,200,400)
topEleJet_all.histoPSet.jetPtBinning = cms.vdouble(0,30,35,37.5,40,50,60,80,120,200,400)
topEleJet_all.histoPSet.jetPtBinning2D = cms.vdouble(0,30,35,40,50,60,80,100,200,400)
topEleJet_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*')
# topEleJet_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu24_v*')


topEleHT_ht = hltTOPmonitoring.clone()
topEleHT_ht.FolderName = cms.string('HLT/TOP/EleHT/HTMonitor')
topEleHT_ht.nmuons = cms.uint32(0)
topEleHT_ht.nelectrons = cms.uint32(1)
topEleHT_ht.njets = cms.uint32(2)
topEleHT_ht.eleSelection = cms.string('pt>50 & abs(eta)<2.1')
topEleHT_ht.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topEleHT_ht.HTcut = cms.double(100)
topEleHT_ht.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1)
topEleHT_ht.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.5,-0.6,0,0.6,1.5,2.1)
topEleHT_ht.histoPSet.elePtBinning = cms.vdouble(0,50,60,80,120,200,400)
topEleHT_ht.histoPSet.elePtBinning2D = cms.vdouble(0,50,70,120,200,400)
topEleHT_ht.histoPSet.jetPtBinning = cms.vdouble(0,30,40,50,60,80,120,200,400)
topEleHT_ht.histoPSet.jetPtBinning2D = cms.vdouble(0,30,40,60,80,100,200,400)
topEleHT_ht.histoPSet.HTBinning  = cms.vdouble(0,100,120,140,150,160,175,200,300,400,500,700)
topEleHT_ht.histoPSet.HTBinning2D  = cms.vdouble(0,100,125,150,175,200,400,700)
topEleHT_ht.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*')
topEleHT_ht.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele35_WPTight_Gsf_v*',
                                                           'HLT_Ele38_WPTight_Gsf_v*',
                                                           'HLT_Ele40_WPTight_Gsf_v*',)

topEleHT_ele = hltTOPmonitoring.clone()
topEleHT_ele.FolderName = cms.string('HLT/TOP/EleHT/ElectronMonitor')
topEleHT_ele.nmuons = cms.uint32(0)
topEleHT_ele.nelectrons = cms.uint32(1)
topEleHT_ele.njets = cms.uint32(2)
topEleHT_ele.eleSelection = cms.string('pt>25 & abs(eta)<2.1')
topEleHT_ele.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topEleHT_ele.HTcut = cms.double(200)
topEleHT_ele.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1)
topEleHT_ele.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.5,-0.6,0,0.6,1.5,2.1)
topEleHT_ele.histoPSet.elePtBinning = cms.vdouble(0,25,30,32.5,35,40,45,50,60,80,120,200,400)
topEleHT_ele.histoPSet.elePtBinning2D = cms.vdouble(0,25,30,40,50,60,80,100,200,400)
topEleHT_ele.histoPSet.jetPtBinning = cms.vdouble(0,30,40,50,60,80,120,200,400)
topEleHT_ele.histoPSet.jetPtBinning2D = cms.vdouble(0,30,40,60,80,100,200,400)
topEleHT_ele.histoPSet.HTBinning  = cms.vdouble(0,200,250,300,350,400,500,700)
topEleHT_ele.histoPSet.HTBinning2D  = cms.vdouble(0,200,250,300,400,500,700)
topEleHT_ele.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*')
topEleHT_ele.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT250_v*',
                                                               'HLT_PFHT370_v*',
                                                               'HLT_PFHT430_v*',
                                                               'HLT_PFHT510_v*',
                                                               'HLT_PFHT590_v*',
                                                               'HLT_PFHT680_v*',
                                                               'HLT_PFHT780_v*',
                                                               'HLT_PFHT890_v*',)

topEleHT_all = hltTOPmonitoring.clone()
topEleHT_all.FolderName = cms.string('HLT/TOP/EleHT/GlobalMonitor')
topEleHT_all.nmuons = cms.uint32(0)
topEleHT_all.nelectrons = cms.uint32(1)
topEleHT_all.njets = cms.uint32(2)
topEleHT_all.eleSelection = cms.string('pt>25 & abs(eta)<2.1')
topEleHT_all.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topEleHT_all.HTcut = cms.double(100)
topEleHT_all.histoPSet.eleEtaBinning = cms.vdouble(-2.1,-1.5,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.5,2.1)
topEleHT_all.histoPSet.eleEtaBinning2D = cms.vdouble(-2.1,-1.5,-0.6,0,0.6,1.5,2.1)
topEleHT_all.histoPSet.elePtBinning = cms.vdouble(0,25,30,32.5,35,40,45,50,60,80,120,200,400)
topEleHT_all.histoPSet.elePtBinning2D = cms.vdouble(0,25,30,40,50,60,80,100,200,400)
topEleHT_all.histoPSet.jetPtBinning = cms.vdouble(0,30,40,50,60,80,120,200,400)
topEleHT_all.histoPSet.jetPtBinning2D = cms.vdouble(0,30,40,60,80,100,200,400)
topEleHT_all.histoPSet.HTBinning  = cms.vdouble(0,100,120,140,150,160,175,200,300,400,500,700)
topEleHT_all.histoPSet.HTBinning2D  = cms.vdouble(0,100,125,150.175,200,400,700)
topEleHT_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*')
# topEleHT_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu24_v*')


#ATHER
#########SingleMuon
topSingleMuonHLTMonitor_Mu24 = hltTOPmonitoring.clone()
topSingleMuonHLTMonitor_Mu24.FolderName = cms.string('HLT/TOP/SingleLepton/SingleMuon/Mu24/')
topSingleMuonHLTMonitor_Mu24.nmuons = cms.uint32(1)
topSingleMuonHLTMonitor_Mu24.nelectrons = cms.uint32(0)
topSingleMuonHLTMonitor_Mu24.njets = cms.uint32(0)
topSingleMuonHLTMonitor_Mu24.eleSelection = cms.string('pt>30 & abs(eta)<2.4')
topSingleMuonHLTMonitor_Mu24.muoSelection = cms.string('pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15')
topSingleMuonHLTMonitor_Mu24.jetSelection = cms.string('pt>20 & abs(eta)<2.4')
topSingleMuonHLTMonitor_Mu24.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu24_eta2p1_v*', 'HLT_IsoMu24_v*')

topSingleMuonHLTMonitor_Mu27 = hltTOPmonitoring.clone()
topSingleMuonHLTMonitor_Mu27.FolderName = cms.string('HLT/TOP/SingleLepton/SingleMuon/Mu27/')
topSingleMuonHLTMonitor_Mu27.nmuons = cms.uint32(1)
topSingleMuonHLTMonitor_Mu27.nelectrons = cms.uint32(0)
topSingleMuonHLTMonitor_Mu27.njets = cms.uint32(0)
topSingleMuonHLTMonitor_Mu27.eleSelection = cms.string('pt>30 & abs(eta)<2.4')
topSingleMuonHLTMonitor_Mu27.muoSelection = cms.string('pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15')
topSingleMuonHLTMonitor_Mu27.jetSelection = cms.string('pt>20 & abs(eta)<2.4')
topSingleMuonHLTMonitor_Mu27.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v*')


topSingleMuonHLTMonitor_Mu50 = hltTOPmonitoring.clone()
topSingleMuonHLTMonitor_Mu50.FolderName = cms.string('HLT/TOP/SingleLepton/SingleMuon/Mu50/')
topSingleMuonHLTMonitor_Mu50.nmuons = cms.uint32(1)
topSingleMuonHLTMonitor_Mu50.nelectrons = cms.uint32(0)
topSingleMuonHLTMonitor_Mu50.njets = cms.uint32(0)
topSingleMuonHLTMonitor_Mu50.eleSelection = cms.string('pt>30 & abs(eta)<2.4')
topSingleMuonHLTMonitor_Mu50.muoSelection = cms.string('pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15')
topSingleMuonHLTMonitor_Mu50.jetSelection = cms.string('pt>20 & abs(eta)<2.4')
topSingleMuonHLTMonitor_Mu50.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu50_v*')


#########DiElectron
topDiElectronHLTMonitor = hltTOPmonitoring.clone()
topDiElectronHLTMonitor.FolderName = cms.string('HLT/TOP/DiLepton/DiElectron/Ele23Ele12/')
topDiElectronHLTMonitor.nmuons = cms.uint32(0)
topDiElectronHLTMonitor.nelectrons = cms.uint32(2)
topDiElectronHLTMonitor.njets = cms.uint32(0)
topDiElectronHLTMonitor.eleSelection = cms.string('pt>15 & abs(eta)<2.4')
topDiElectronHLTMonitor.muoSelection = cms.string('pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')
topDiElectronHLTMonitor.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiElectronHLTMonitor.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*')

topDiElectronHLTMonitor_Dz = topDiElectronHLTMonitor.clone()
topDiElectronHLTMonitor_Dz.FolderName = cms.string('HLT/TOP/DiLepton/DiElectron/Ele23Ele12_DzEfficiency/')
topDiElectronHLTMonitor_Dz.denGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v*'])

#########DiMuon
topDiMuonHLTMonitor_noDz = hltTOPmonitoring.clone()
topDiMuonHLTMonitor_noDz.FolderName = cms.string('HLT/TOP/DiLepton/DiMuon/Mu17_Mu8/')
topDiMuonHLTMonitor_noDz.nmuons = cms.uint32(2)
topDiMuonHLTMonitor_noDz.nelectrons = cms.uint32(0)
topDiMuonHLTMonitor_noDz.njets = cms.uint32(0)
topDiMuonHLTMonitor_noDz.eleSelection = cms.string('pt>15 & abs(eta)<2.4')
topDiMuonHLTMonitor_noDz.muoSelection = cms.string('pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')
topDiMuonHLTMonitor_noDz.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiMuonHLTMonitor_noDz.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*')

topDiMuonHLTMonitor_Dz = hltTOPmonitoring.clone()
topDiMuonHLTMonitor_Dz.FolderName = cms.string('HLT/TOP/DiLepton/DiMuon/Mu17_Mu8_Dz/')
topDiMuonHLTMonitor_Dz.nmuons = cms.uint32(2)
topDiMuonHLTMonitor_Dz.nelectrons = cms.uint32(0)
topDiMuonHLTMonitor_Dz.njets = cms.uint32(0)
topDiMuonHLTMonitor_Dz.eleSelection = cms.string('pt>15 & abs(eta)<2.4')              
topDiMuonHLTMonitor_Dz.muoSelection = cms.string('pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')
topDiMuonHLTMonitor_Dz.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiMuonHLTMonitor_Dz.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')

topDiMuonHLTMonitor_Dz_Mu17_Mu8 = topDiMuonHLTMonitor_Dz.clone()
topDiMuonHLTMonitor_Dz_Mu17_Mu8.FolderName = cms.string('HLT/TOP/DiLepton/DiMuon/Mu17_Mu8_DzEfficiency/')
topDiMuonHLTMonitor_Dz_Mu17_Mu8.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')
topDiMuonHLTMonitor_Dz_Mu17_Mu8.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*')

topDiMuonHLTMonitor_Mass8 = hltTOPmonitoring.clone()
topDiMuonHLTMonitor_Mass8.FolderName = cms.string('HLT/TOP/DiLepton/DiMuon/Mass8/')
topDiMuonHLTMonitor_Mass8.nmuons = cms.uint32(2)
topDiMuonHLTMonitor_Mass8.nelectrons = cms.uint32(0)
topDiMuonHLTMonitor_Mass8.njets = cms.uint32(0)
topDiMuonHLTMonitor_Mass8.eleSelection = cms.string('pt>15 & abs(eta)<2.4')
topDiMuonHLTMonitor_Mass8.muoSelection = cms.string('pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')
topDiMuonHLTMonitor_Mass8.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiMuonHLTMonitor_Mass8.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v*')


topDiMuonHLTMonitor_Mass3p8 = hltTOPmonitoring.clone()
#topDiMuonHLTMonitor_Mass3p8.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/DiLepton/DiMuon/Mass3p8/')
topDiMuonHLTMonitor_Mass3p8.FolderName = cms.string('HLT/TOP/DiLepton/DiMuon/Mass3p8/')
topDiMuonHLTMonitor_Mass3p8.nmuons = cms.uint32(2)
topDiMuonHLTMonitor_Mass3p8.nelectrons = cms.uint32(0)
topDiMuonHLTMonitor_Mass3p8.njets = cms.uint32(0)
topDiMuonHLTMonitor_Mass3p8.eleSelection = cms.string('pt>15 & abs(eta)<2.4')
topDiMuonHLTMonitor_Mass3p8.muoSelection = cms.string('pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')
topDiMuonHLTMonitor_Mass3p8.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiMuonHLTMonitor_Mass3p8.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v*')

topDiMuonHLTMonitor_Mass8Mon = hltTOPmonitoring.clone()
#topDiMuonHLTMonitor_Mass8Mon.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/DiLepton/DiMuon/Mu17_Mu8_Mass8Efficiency/')
topDiMuonHLTMonitor_Mass8Mon.FolderName = cms.string('HLT/TOP/DiLepton/DiMuon/Mu17_Mu8_Mass8Efficiency/')
topDiMuonHLTMonitor_Mass8Mon.nmuons = cms.uint32(2)
topDiMuonHLTMonitor_Mass8Mon.nelectrons = cms.uint32(0)
topDiMuonHLTMonitor_Mass8Mon.njets = cms.uint32(0)
topDiMuonHLTMonitor_Mass8Mon.eleSelection = cms.string('pt>15 & abs(eta)<2.4')
topDiMuonHLTMonitor_Mass8Mon.muoSelection = cms.string('pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')
topDiMuonHLTMonitor_Mass8Mon.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiMuonHLTMonitor_Mass8Mon.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v*')
topDiMuonHLTMonitor_Mass8Mon.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')

topDiMuonHLTMonitor_Mass3p8Mon = hltTOPmonitoring.clone()
#topDiMuonHLTMonitor_Mass3p8Mon.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/DiLepton/DiMuon/Mu17_Mu8_Mass3p8Efficiency/')
topDiMuonHLTMonitor_Mass3p8Mon.FolderName = cms.string('HLT/TOP/DiLepton/DiMuon/Mu17_Mu8_Mass3p8Efficiency/')
topDiMuonHLTMonitor_Mass3p8Mon.nmuons = cms.uint32(2)
topDiMuonHLTMonitor_Mass3p8Mon.nelectrons = cms.uint32(0)
topDiMuonHLTMonitor_Mass3p8Mon.njets = cms.uint32(0)
topDiMuonHLTMonitor_Mass3p8Mon.eleSelection = cms.string('pt>15 & abs(eta)<2.4')
topDiMuonHLTMonitor_Mass3p8Mon.muoSelection = cms.string('pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)')
topDiMuonHLTMonitor_Mass3p8Mon.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topDiMuonHLTMonitor_Mass3p8Mon.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v*')
topDiMuonHLTMonitor_Mass3p8Mon.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')

#########ElecMuon
topElecMuonHLTMonitor = hltTOPmonitoring.clone()
topElecMuonHLTMonitor.FolderName = cms.string('HLT/TOP/DiLepton/ElecMuon/OR/')
topElecMuonHLTMonitor.nmuons = cms.uint32(1)
topElecMuonHLTMonitor.nelectrons = cms.uint32(1)
topElecMuonHLTMonitor.njets = cms.uint32(0)
topElecMuonHLTMonitor.eleSelection = cms.string('pt>15 & abs(eta)<2.4')
topElecMuonHLTMonitor.muoSelection = cms.string('pt>15 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25  & isPFMuon & (isTrackerMuon || isGlobalMuon)') 
topElecMuonHLTMonitor.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
topElecMuonHLTMonitor.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*','HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*', 'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*','HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*','HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*', 'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v*')

#DZ monitor
topElecMuonHLTMonitor_Dz_Mu12Ele23 = topElecMuonHLTMonitor.clone()
topElecMuonHLTMonitor_Dz_Mu12Ele23.FolderName = cms.string('HLT/TOP/DiLepton/ElecMuon/Mu12Ele23_DzEfficiency/')
topElecMuonHLTMonitor_Dz_Mu12Ele23.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*')
topElecMuonHLTMonitor_Dz_Mu12Ele23.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*')

topElecMuonHLTMonitor_Dz_Mu8Ele23 = topElecMuonHLTMonitor.clone()
topElecMuonHLTMonitor_Dz_Mu8Ele23.FolderName = cms.string('HLT/TOP/DiLepton/ElecMuon/Mu8Ele23_DzEfficiency/')
topElecMuonHLTMonitor_Dz_Mu8Ele23.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*')
topElecMuonHLTMonitor_Dz_Mu8Ele23.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*')

topElecMuonHLTMonitor_Dz_Mu23Ele12 = topElecMuonHLTMonitor.clone()
topElecMuonHLTMonitor_Dz_Mu23Ele12.FolderName = cms.string('HLT/TOP/DiLepton/ElecMuon/Mu23Ele12_DzEfficiency/')
topElecMuonHLTMonitor_Dz_Mu23Ele12.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*')
topElecMuonHLTMonitor_Dz_Mu23Ele12.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v*')

#individual paths
topElecMuonHLTMonitor_Mu12Ele23 = topElecMuonHLTMonitor.clone()
topElecMuonHLTMonitor_Mu12Ele23.FolderName = cms.string('HLT/TOP/DiLepton/ElecMuon/Mu12Ele23/')
topElecMuonHLTMonitor_Mu12Ele23.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*')

topElecMuonHLTMonitor_Mu8Ele23 = topElecMuonHLTMonitor.clone()
topElecMuonHLTMonitor_Mu8Ele23.FolderName = cms.string('HLT/TOP/DiLepton/ElecMuon/Mu8Ele23/')
topElecMuonHLTMonitor_Mu8Ele23.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*')

topElecMuonHLTMonitor_Mu23Ele12 = topElecMuonHLTMonitor.clone()
topElecMuonHLTMonitor_Mu23Ele12.FolderName = cms.string('HLT/TOP/DiLepton/ElecMuon/Mu23Ele12/')
topElecMuonHLTMonitor_Mu23Ele12.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*')

#reference paths
topElecMuonHLTMonitor_Mu12Ele23_ref = topElecMuonHLTMonitor.clone()
topElecMuonHLTMonitor_Mu12Ele23_ref.FolderName = cms.string('HLT/TOP/DiLepton/ElecMuon/Mu12Ele23_Ref/')
topElecMuonHLTMonitor_Mu12Ele23_ref.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*')

topElecMuonHLTMonitor_Mu8Ele23_ref = topElecMuonHLTMonitor.clone()
topElecMuonHLTMonitor_Mu8Ele23_ref.FolderName = cms.string('HLT/TOP/DiLepton/ElecMuon/Mu8Ele23_Ref/')
topElecMuonHLTMonitor_Mu8Ele23_ref.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*')

topElecMuonHLTMonitor_Mu23Ele12_ref = topElecMuonHLTMonitor.clone()
topElecMuonHLTMonitor_Mu23Ele12_ref.FolderName = cms.string('HLT/TOP/DiLepton/ElecMuon/Mu23Ele12_Ref/')
topElecMuonHLTMonitor_Mu23Ele12_ref.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v*')


# Marina

fullyhadronic_ref350 = hltTOPmonitoring.clone()
fullyhadronic_ref350.FolderName = cms.string('HLT/TOP/FullyHadronic/Reference/PFHT350Monitor/')
# Selections
fullyhadronic_ref350.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_ref350.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_ref350.HTcut            = cms.double(250)
# Binning
fullyhadronic_ref350.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_ref350.histoPSet.HTBinning = cms.vdouble(0,240,260,280,300,320,340,360,380,400,420,440,460,480,
                                                       500,520,540,560,580,600,650,700,750,800,850,900,1000)
# Trigger
fullyhadronic_ref350.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT350_v*')
fullyhadronic_ref350.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v*')


fullyhadronic_ref370 = hltTOPmonitoring.clone()
fullyhadronic_ref370.FolderName = cms.string('HLT/TOP/FullyHadronic/Reference/PFHT370Monitor/')
# Selections
fullyhadronic_ref370.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_ref370.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_ref370.HTcut            = cms.double(250)
# Binning
fullyhadronic_ref370.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_ref370.histoPSet.HTBinning = cms.vdouble(0,240,260,280,300,320,340,360,380,400,420,440,460,480,
                                                       500,520,540,560,580,600,650,700,750,800,850,900,1000)
# Trigger
fullyhadronic_ref370.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT370_v*')
fullyhadronic_ref370.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v*')

fullyhadronic_ref430 = hltTOPmonitoring.clone()
fullyhadronic_ref430.FolderName = cms.string('HLT/TOP/FullyHadronic/Reference/PFHT430Monitor/')
# Selections
fullyhadronic_ref430.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_ref430.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_ref430.HTcut            = cms.double(250)
# Binning
fullyhadronic_ref430.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_ref430.histoPSet.HTBinning = cms.vdouble(0,240,260,280,300,320,340,360,380,400,420,440,460,480,
                                                       500,520,540,560,580,600,650,700,750,800,850,900,1000)
# Trigger
fullyhadronic_ref430.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT430_v*')
fullyhadronic_ref430.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v*')


fullyhadronic_DoubleBTag_all = hltTOPmonitoring.clone()
fullyhadronic_DoubleBTag_all.FolderName   = cms.string('HLT/TOP/FullyHadronic/DoubleBTag/GlobalMonitor/')
# Selections
fullyhadronic_DoubleBTag_all.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_DoubleBTag_all.njets            = cms.uint32(6)
fullyhadronic_DoubleBTag_all.jetSelection     = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_all.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_all.HTcut            = cms.double(500)
fullyhadronic_DoubleBTag_all.nbjets           = cms.uint32(2)
fullyhadronic_DoubleBTag_all.bjetSelection    = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_all.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_DoubleBTag_all.workingpoint     = cms.double(0.4941) #  Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)

# Binning
fullyhadronic_DoubleBTag_all.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_DoubleBTag_all.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200)
fullyhadronic_DoubleBTag_all.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers 
fullyhadronic_DoubleBTag_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94_v*')
fullyhadronic_DoubleBTag_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v*')

fullyhadronic_DoubleBTag_jet = hltTOPmonitoring.clone()
fullyhadronic_DoubleBTag_jet.FolderName   = cms.string('HLT/TOP/FullyHadronic/DoubleBTag/JetMonitor/')
# Selections
fullyhadronic_DoubleBTag_jet.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_DoubleBTag_jet.njets            = cms.uint32(6)
fullyhadronic_DoubleBTag_jet.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_jet.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_jet.HTcut            = cms.double(500)
fullyhadronic_DoubleBTag_jet.nbjets           = cms.uint32(2)
fullyhadronic_DoubleBTag_jet.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_jet.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_DoubleBTag_jet.workingpoint     = cms.double(0.4941) #  Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)

# Binning 
fullyhadronic_DoubleBTag_jet.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_DoubleBTag_jet.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200)
fullyhadronic_DoubleBTag_jet.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_DoubleBTag_jet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT400_SixPFJet32_v*')
fullyhadronic_DoubleBTag_jet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT370_v*')

fullyhadronic_DoubleBTag_bjet = hltTOPmonitoring.clone()
fullyhadronic_DoubleBTag_bjet.FolderName   = cms.string('HLT/TOP/FullyHadronic/DoubleBTag/BJetMonitor/')
# Selections
fullyhadronic_DoubleBTag_bjet.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_DoubleBTag_bjet.njets            = cms.uint32(6)
fullyhadronic_DoubleBTag_bjet.jetSelection     = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_bjet.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_bjet.HTcut            = cms.double(500)
fullyhadronic_DoubleBTag_bjet.nbjets           = cms.uint32(2)
fullyhadronic_DoubleBTag_bjet.bjetSelection    = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_bjet.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_DoubleBTag_bjet.workingpoint     = cms.double(0.1522) # Loose (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
# Binning
fullyhadronic_DoubleBTag_bjet.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_DoubleBTag_bjet.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200)
fullyhadronic_DoubleBTag_bjet.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_DoubleBTag_bjet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94_v*')
fullyhadronic_DoubleBTag_bjet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT400_SixPFJet32_v*')

fullyhadronic_DoubleBTag_ref = hltTOPmonitoring.clone()
fullyhadronic_DoubleBTag_ref.FolderName   = cms.string('HLT/TOP/FullyHadronic/DoubleBTag/RefMonitor/')
# Selections
fullyhadronic_DoubleBTag_ref.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_DoubleBTag_ref.njets            = cms.uint32(6)
fullyhadronic_DoubleBTag_ref.jetSelection     = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_ref.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_ref.HTcut            = cms.double(500)
fullyhadronic_DoubleBTag_ref.nbjets           = cms.uint32(0)
fullyhadronic_DoubleBTag_ref.bjetSelection    = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_DoubleBTag_ref.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_DoubleBTag_ref.workingpoint     = cms.double(0.4941) # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
# Binning
fullyhadronic_DoubleBTag_ref.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_DoubleBTag_ref.histoPSet.jetPtBinning = cms.vdouble(0,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200)
fullyhadronic_DoubleBTag_ref.histoPSet.HTBinning    = cms.vdouble(0,360,380,400,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_DoubleBTag_ref.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT400_SixPFJet32_v*')
fullyhadronic_DoubleBTag_ref.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v*')

fullyhadronic_SingleBTag_all = hltTOPmonitoring.clone()
fullyhadronic_SingleBTag_all.FolderName= cms.string('HLT/TOP/FullyHadronic/SingleBTag/GlobalMonitor/')
# Selections
fullyhadronic_SingleBTag_all.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_SingleBTag_all.njets            = cms.uint32(6)
fullyhadronic_SingleBTag_all.jetSelection     = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_SingleBTag_all.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_all.HTcut            = cms.double(500)
fullyhadronic_SingleBTag_all.nbjets           = cms.uint32(2)
fullyhadronic_SingleBTag_all.bjetSelection    = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_SingleBTag_all.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_SingleBTag_all.workingpoint     = cms.double(0.4941) # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
# Binning
fullyhadronic_SingleBTag_all.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_SingleBTag_all.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200)
fullyhadronic_SingleBTag_all.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_SingleBTag_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59_v*')
fullyhadronic_SingleBTag_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v*')

fullyhadronic_SingleBTag_jet = hltTOPmonitoring.clone()
fullyhadronic_SingleBTag_jet.FolderName= cms.string('HLT/TOP/FullyHadronic/SingleBTag/JetMonitor/')
# Selection
fullyhadronic_SingleBTag_jet.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_SingleBTag_jet.njets            = cms.uint32(6)
fullyhadronic_SingleBTag_jet.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_jet.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_jet.HTcut            = cms.double(500)
fullyhadronic_SingleBTag_jet.nbjets           = cms.uint32(2)
fullyhadronic_SingleBTag_jet.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_jet.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_SingleBTag_jet.workingpoint     = cms.double(0.4941) # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
# Binning
fullyhadronic_SingleBTag_jet.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_SingleBTag_jet.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200)
fullyhadronic_SingleBTag_jet.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_SingleBTag_jet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT450_SixPFJet36_v*')
fullyhadronic_SingleBTag_jet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT430_v*')

fullyhadronic_SingleBTag_bjet = hltTOPmonitoring.clone()
fullyhadronic_SingleBTag_bjet.FolderName= cms.string('HLT/TOP/FullyHadronic/SingleBTag/BJetMonitor/')
# Selection
fullyhadronic_SingleBTag_bjet.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_SingleBTag_bjet.njets            = cms.uint32(6)
fullyhadronic_SingleBTag_bjet.jetSelection     = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_SingleBTag_bjet.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_bjet.HTcut            = cms.double(500)
fullyhadronic_SingleBTag_bjet.nbjets           = cms.uint32(2)
fullyhadronic_SingleBTag_bjet.bjetSelection    = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_SingleBTag_bjet.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_SingleBTag_bjet.workingpoint     = cms.double(0.1522) # Loose (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
# Binning
fullyhadronic_SingleBTag_bjet.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_SingleBTag_bjet.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200)
fullyhadronic_SingleBTag_bjet.histoPSet.HTBinning    = cms.vdouble(0,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_SingleBTag_bjet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59_v*')
fullyhadronic_SingleBTag_bjet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT450_SixPFJet36_v*')

fullyhadronic_SingleBTag_ref = hltTOPmonitoring.clone()
fullyhadronic_SingleBTag_ref.FolderName= cms.string('HLT/TOP/FullyHadronic/SingleBTag/RefMonitor/')
# Selection
fullyhadronic_SingleBTag_ref.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_SingleBTag_ref.njets            = cms.uint32(6)
fullyhadronic_SingleBTag_ref.jetSelection     = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_SingleBTag_ref.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_SingleBTag_ref.HTcut            = cms.double(500)
fullyhadronic_SingleBTag_ref.nbjets           = cms.uint32(0)
fullyhadronic_SingleBTag_ref.bjetSelection    = cms.string('pt>40 & abs(eta)<2.4')
fullyhadronic_SingleBTag_ref.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_SingleBTag_ref.workingpoint     = cms.double(0.4941) # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
# Binning
fullyhadronic_SingleBTag_ref.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_SingleBTag_ref.histoPSet.jetPtBinning = cms.vdouble(0,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160,200)
fullyhadronic_SingleBTag_ref.histoPSet.HTBinning    = cms.vdouble(0,400,420,440,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_SingleBTag_ref.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT450_SixPFJet36_v*')
fullyhadronic_SingleBTag_ref.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v*')

# TripleBTag
fullyhadronic_TripleBTag_all = hltTOPmonitoring.clone()
fullyhadronic_TripleBTag_all.FolderName   = cms.string('HLT/TOP/FullyHadronic/TripleBTag/GlobalMonitor/')
# Selections
fullyhadronic_TripleBTag_all.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_TripleBTag_all.njets            = cms.uint32(4)
fullyhadronic_TripleBTag_all.jetSelection     = cms.string('pt>45 & abs(eta)<2.4')
fullyhadronic_TripleBTag_all.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_TripleBTag_all.HTcut            = cms.double(500)
fullyhadronic_TripleBTag_all.nbjets           = cms.uint32(4)
fullyhadronic_TripleBTag_all.bjetSelection    = cms.string('pt>45 & abs(eta)<2.4')
fullyhadronic_TripleBTag_all.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_TripleBTag_all.workingpoint     = cms.double(0.4941) # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
# Binning
fullyhadronic_TripleBTag_all.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_TripleBTag_all.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
fullyhadronic_TripleBTag_all.histoPSet.HTBinning    = cms.vdouble(0,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_TripleBTag_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5_v*')
fullyhadronic_TripleBTag_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu27_v*')

fullyhadronic_TripleBTag_jet = hltTOPmonitoring.clone()
fullyhadronic_TripleBTag_jet.FolderName   = cms.string('HLT/TOP/FullyHadronic/TripleBTag/JetMonitor/')
# Selections
fullyhadronic_TripleBTag_jet.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_TripleBTag_jet.njets            = cms.uint32(4)
fullyhadronic_TripleBTag_jet.jetSelection     = cms.string('pt>45 & abs(eta)<2.4')
fullyhadronic_TripleBTag_jet.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_TripleBTag_jet.HTcut            = cms.double(500)
fullyhadronic_TripleBTag_jet.nbjets           = cms.uint32(4)
fullyhadronic_TripleBTag_jet.bjetSelection    = cms.string('pt>45 & abs(eta)<2.4')
fullyhadronic_TripleBTag_jet.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_TripleBTag_jet.workingpoint     = cms.double(0.4941) # Medium (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
# Binning
fullyhadronic_TripleBTag_jet.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_TripleBTag_jet.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
fullyhadronic_TripleBTag_jet.histoPSet.HTBinning    = cms.vdouble(0,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers
fullyhadronic_TripleBTag_jet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v*')
fullyhadronic_TripleBTag_jet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT350_v*')

fullyhadronic_TripleBTag_bjet = hltTOPmonitoring.clone()
fullyhadronic_TripleBTag_bjet.FolderName   = cms.string('HLT/TOP/FullyHadronic/TripleBTag/BJetMonitor/')
# Selections
fullyhadronic_TripleBTag_bjet.leptJetDeltaRmin = cms.double(0.0)
fullyhadronic_TripleBTag_bjet.njets            = cms.uint32(4)
fullyhadronic_TripleBTag_bjet.jetSelection     = cms.string('pt>45 & abs(eta)<2.4')
fullyhadronic_TripleBTag_bjet.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
fullyhadronic_TripleBTag_bjet.HTcut            = cms.double(500)
fullyhadronic_TripleBTag_bjet.nbjets           = cms.uint32(4)
fullyhadronic_TripleBTag_bjet.bjetSelection    = cms.string('pt>45 & abs(eta)<2.4')
fullyhadronic_TripleBTag_bjet.btagalgo         = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll")
fullyhadronic_TripleBTag_bjet.workingpoint     = cms.double(0.1522) # Loose (According to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X)
# Binning
fullyhadronic_TripleBTag_bjet.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
fullyhadronic_TripleBTag_bjet.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
fullyhadronic_TripleBTag_bjet.histoPSet.HTBinning    = cms.vdouble(0,460,480,500,520,540,560,580,600,650,700,750,800,850,900)
# Triggers 
fullyhadronic_TripleBTag_bjet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5_v*')
fullyhadronic_TripleBTag_bjet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v*')


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
    + topDiMuonHLTMonitor_noDz
    + topDiMuonHLTMonitor_Dz
    + topDiMuonHLTMonitor_Mass8
    + topDiMuonHLTMonitor_Mass3p8
    + topDiMuonHLTMonitor_Mass8Mon
    + topDiMuonHLTMonitor_Mass3p8Mon
    + topElecMuonHLTMonitor
    + fullyhadronic_ref350
    + fullyhadronic_ref370
    + fullyhadronic_ref430
    + fullyhadronic_DoubleBTag_all
    + fullyhadronic_DoubleBTag_jet
    + fullyhadronic_DoubleBTag_bjet
    + fullyhadronic_DoubleBTag_ref
    + fullyhadronic_SingleBTag_all
    + fullyhadronic_SingleBTag_jet
    + fullyhadronic_SingleBTag_bjet
    + fullyhadronic_SingleBTag_ref
    + fullyhadronic_TripleBTag_all
    + fullyhadronic_TripleBTag_jet
    + fullyhadronic_TripleBTag_bjet
    + topDiElectronHLTMonitor_Dz
    + topElecMuonHLTMonitor_Dz_Mu12Ele23
    + topElecMuonHLTMonitor_Dz_Mu8Ele23
    + topElecMuonHLTMonitor_Dz_Mu23Ele12
    + topElecMuonHLTMonitor_Mu12Ele23
    + topElecMuonHLTMonitor_Mu8Ele23
    + topElecMuonHLTMonitor_Mu23Ele12
    + topElecMuonHLTMonitor_Mu12Ele23_ref
    + topElecMuonHLTMonitor_Mu8Ele23_ref
    + topElecMuonHLTMonitor_Mu23Ele12_ref
    + topDiMuonHLTMonitor_Dz_Mu17_Mu8,
     cms.Task(egmGsfElectronIDsForDQM) # Use of electron VID requires this module being executed first
)

topHLTDQMSourceExtra = cms.Sequence(
)
