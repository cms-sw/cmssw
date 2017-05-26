import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

dummy = hltTOPmonitoring.clone()
dummy.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/dummy/')
dummy.nmuons = cms.uint32(0)
dummy.nelectrons = cms.uint32(1)
dummy.njets = cms.uint32(2)
dummy.eleSelection = cms.string('pt>15 & abs(eta)<2.4')
dummy.jetSelection = cms.string('pt>20 & abs(eta)<2.4')
dummy.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned')
dummy.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned')

test = hltTOPmonitoring.clone()
test.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/test/')
test.nmuons = cms.uint32(0)
test.nelectrons = cms.uint32(1)
test.njets = cms.uint32(2)
test.eleSelection = cms.string('pt>15 & abs(eta)<2.4')
test.jetSelection = cms.string('pt>20 & abs(eta)<2.4')
test.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned')
test.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_v*')

topMonitorHLT = cms.Sequence(
    dummy
    + test
)
