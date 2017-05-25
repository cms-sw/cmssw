import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

dummy = hltTOPmonitoring.clone()
dummy.FolderName = cms.string('HLT/TOP/dummy/')
dummy.nmuons = cms.int32(0)
dummy.nelectrons = cms.int32(1)
dummy.njets = cms.int32(1)
dummy.eleSelection = cms.string('pt>15 & abs(eta)<2.4')
dummy.jetSelection = cms.string('pt>20 & abs(eta)<2.4')
dummy.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned')
dummy.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned')


topMonitorHLT = cms.Sequence(
    dummy
)
