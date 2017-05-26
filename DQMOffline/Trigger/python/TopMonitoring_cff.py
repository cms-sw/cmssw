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


topMonitorHLT = cms.Sequence(

    eleJet_ele
    + eleJet_jet
    + eleJet_all
    + eleHT_ele
    + eleHT_ht
    + eleHT_all

)
