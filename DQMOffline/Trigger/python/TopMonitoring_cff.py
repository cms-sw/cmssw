import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

eleJet_jet = hltTOPmonitoring.clone()
eleJet_jet.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleJet/JetMonitor')
eleJet_jet.nmuons = cms.uint32(0)
eleJet_jet.nelectrons = cms.uint32(1)
eleJet_jet.njets = cms.uint32(2)
eleJet_jet.eleSelection = cms.string('pt>50 & abs(eta)<2.1')
eleJet_jet.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
eleJet_jet.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*')
eleJet_jet.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_v*',
                                                             'HLT_Ele35_WPTight_Gsf_v*',
                                                             'HLT_Ele38_WPTight_Gsf_v*',
                                                             'HLT_Ele40_WPTight_Gsf_v*',)

eleJet_ele = hltTOPmonitoring.clone()
eleJet_ele.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleJet/ElectronMonitor')
eleJet_ele.nmuons = cms.uint32(0)
eleJet_ele.nelectrons = cms.uint32(1)
eleJet_ele.njets = cms.uint32(2)
eleJet_ele.eleSelection = cms.string('pt>25 & abs(eta)<2.1')
eleJet_ele.jetSelection = cms.string('pt>50 & abs(eta)<2.4')
eleJet_ele.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*')
eleJet_ele.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet60_v*')

eleJet_all = hltTOPmonitoring.clone()
eleJet_all.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleJet/GlobalMonitor')
eleJet_all.nmuons = cms.uint32(0)
eleJet_all.nelectrons = cms.uint32(1)
eleJet_all.njets = cms.uint32(2)
eleJet_all.eleSelection = cms.string('pt>25 & abs(eta)<2.1')
eleJet_all.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
eleJet_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v*')
# eleJet_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu24_v*')

eleHT_ht = hltTOPmonitoring.clone()
eleHT_ht.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleHT/HTMonitor')
eleHT_ht.nmuons = cms.uint32(0)
eleHT_ht.nelectrons = cms.uint32(1)
eleHT_ht.njets = cms.uint32(2)
eleHT_ht.eleSelection = cms.string('pt>50 & abs(eta)<2.1')
eleHT_ht.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
eleHT_ht.HTcut = cms.double(200)
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
eleHT_ele.eleSelection = cms.string('pt>25 & abs(eta)<2.1')
eleHT_ele.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
eleHT_ele.HTcut = cms.double(200)
eleHT_ele.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*')
eleHT_ele.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_HT200_v*',
                                                            'HLT_HT275_v*',)

eleHT_all = hltTOPmonitoring.clone()
eleHT_all.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/EleHT/GlobalMonitor')
eleHT_all.nmuons = cms.uint32(0)
eleHT_all.nelectrons = cms.uint32(1)
eleHT_all.njets = cms.uint32(2)
eleHT_all.eleSelection = cms.string('pt>25 & abs(eta)<2.1')
eleHT_all.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
eleHT_all.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v*')
# eleHT_all.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_IsoMu24_v*')


#ATHER
topSingleMuonHLTValidation = hltTOPmonitoring.clone()
topSingleMuonHLTValidation.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/Top/SingleLepton/SingleMuon/')
topSingleMuonHLTValidation.nmuons = cms.uint32(1)
topSingleMuonHLTValidation.nelectrons = cms.uint32(0)
topSingleMuonHLTValidation.njets = cms.uint32(4)
topSingleMuonHLTValidation.eleSelection = cms.string('pt>30 & abs(eta)<2.5 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')
topSingleMuonHLTValidation.muoSelection = cms.string('pt>26 & abs(eta)<2.1 & (pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt)/pt < 0.12')
topSingleMuonHLTValidation.jetSelection = cms.string('pt>20 & abs(eta)<2.5')
topSingleMuonHLTValidation.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Mu20_v*', 'HLT_TkMu20_v*' , 'HLT_Mu27_v*', 'HLT_TkMu27_v*', 'HLT_TkMu50_v*', 'HLT_Mu50_v*', 'HLT_IsoMu24_eta2p1_v*', 'HLT_IsoMu24_v*', 'HLT_IsoMu27_v*', 'HLT_IsoMu20_v*', 'HLT_IsoTkMu24_eta2p1_v*', 'HLT_IsoTkMu24_v*', 'HLT_IsoTkMu27_v*', 'HLT_IsoTkMu20_v*'])



topDiElectronHLTValidation = hltTOPmonitoring.clone()
topDiElectronHLTValidation.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/Top/DiLepton/DiElectron/')
topDiElectronHLTValidation.nmuons = cms.uint32(0)
topDiElectronHLTValidation.nelectrons = cms.uint32(2)
topDiElectronHLTValidation.njets = cms.uint32(2)
topDiElectronHLTValidation.eleSelection = cms.string('pt>20 & abs(eta)<2.5  & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')
topDiElectronHLTValidation.muoSelection = cms.string('pt>20 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt)/pt < 0.12')   
topDiElectronHLTValidation.jetSelection = cms.string('pt>30 & abs(eta)<2.5')
topDiElectronHLTValidation.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Ele12_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*'])



topDiMuonHLTValidation = hltTOPmonitoring.clone()
topDiMuonHLTValidation.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/Top/DiLepton/DiMuon/')
topDiMuonHLTValidation.nmuons = cms.uint32(2)
topDiMuonHLTValidation.nelectrons = cms.uint32(0)
topDiMuonHLTValidation.njets = cms.uint32(2)
topDiMuonHLTValidation.eleSelection = cms.string('pt>20 & abs(eta)<2.5  & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')              
topDiMuonHLTValidation.muoSelection = cms.string('pt>20 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt)/pt < 0.12')  
topDiMuonHLTValidation.jetSelection = cms.string('pt>30 & abs(eta)<2.5')
topDiMuonHLTValidation.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*','HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*'])



topElecMuonHLTValidation = hltTOPmonitoring.clone()
topElecMuonHLTValidation.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/Top/DiLepton/ElecMuon/')
topElecMuonHLTValidation.nmuons = cms.uint32(1)
topElecMuonHLTValidation.nelectrons = cms.uint32(1)
topElecMuonHLTValidation.njets = cms.uint32(2)
topElecMuonHLTValidation.eleSelection = cms.string('pt>20 & abs(eta)<2.5 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')
topElecMuonHLTValidation.muoSelection = cms.string('pt>20 & abs(eta)<2.4 & (pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt)/pt < 0.12')           
topElecMuonHLTValidation.jetSelection = cms.string('pt>30 & abs(eta)<2.5')
topElecMuonHLTValidation.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*','HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*'])



singleTopSingleMuonHLTValidation = hltTOPmonitoring.clone()
singleTopSingleMuonHLTValidation.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/SingleTop/SingleMuon/')
singleTopSingleMuonHLTValidation.nmuons = cms.uint32(1)
singleTopSingleMuonHLTValidation.nelectrons = cms.uint32(0)
singleTopSingleMuonHLTValidation.njets = cms.uint32(2)
singleTopSingleMuonHLTValidation.eleSelection = cms.string('pt>30 & abs(eta)<2.5 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt < 0.1')   
singleTopSingleMuonHLTValidation.muoSelection = cms.string('pt>26 & abs(eta)<2.1 & (pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt)/pt < 0.12')
singleTopSingleMuonHLTValidation.jetSelection = cms.string('pt>40 & abs(eta)<5.0')
singleTopSingleMuonHLTValidation.numGenericTriggerEventPSet.hltPaths = cms.vstring(['HLT_Mu20_v*', 'HLT_TkMu20_v*' , 'HLT_Mu27_v*', 'HLT_TkMu27_v*', 'HLT_TkMu50_v*', 'HLT_Mu50_v*', 'HLT_IsoMu24_eta2p1_v*', 'HLT_IsoMu24_v*', 'HLT_IsoMu27_v*', 'HLT_IsoMu20_v*', 'HLT_IsoTkMu24_eta2p1_v*', 'HLT_IsoTkMu24_v*', 'HLT_IsoTkMu27_v*', 'HLT_IsoTkMu20_v*'])



topMonitorHLT = cms.Sequence(
    eleJet_ele
    + eleJet_jet
    + eleJet_all
    + eleHT_ele
    + eleHT_ht
    + eleHT_all
    + topSingleMuonHLTValidation
    + topDiElectronHLTValidation
    + topDiMuonHLTValidation
    + topElecMuonHLTValidation
    + singleTopSingleMuonHLTValidation

    )
