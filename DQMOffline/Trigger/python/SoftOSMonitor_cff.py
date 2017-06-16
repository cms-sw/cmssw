import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SoftOSMonitor_cfi import hltSOSmonitoring

# HLT_PFMETNoMu90_PFMHTNoMu90_IDTight
DoubleMu3_PFMET50_DZ_PFMHT60_METmonitoring = hltSOSmonitoring.clone()
DoubleMu3_PFMET50_DZ_PFMHT60_METmonitoring.FolderName = cms.string('HLT/SOS/MET/')
DoubleMu3_PFMET50_DZ_PFMHT60_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu3_PFMET50_DZ_PFMHT60_v1")
DoubleMu3_PFMET50_DZ_PFMHT60_METmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v7")
DoubleMu3_PFMET50_DZ_PFMHT60_METmonitoring.turn_on = cms.string("met")
DoubleMu3_PFMET50_DZ_PFMHT60_METmonitoring.mu2_pt_cut = cms.int32(20)

DoubleMu3_PFMET50_DZ_PFMHT60_MUmonitoring = hltSOSmonitoring.clone()
DoubleMu3_PFMET50_DZ_PFMHT60_MUmonitoring.FolderName = cms.string('HLT/SOS/MU/')
DoubleMu3_PFMET50_DZ_PFMHT60_MUmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu3_PFMET50_DZ_PFMHT60_v1")
DoubleMu3_PFMET50_DZ_PFMHT60_MUmonitoring.turn_on = cms.string("mu")
DoubleMu3_PFMET50_DZ_PFMHT60_MUmonitoring.met_pt_cut = cms.int32(200)
DoubleMu3_PFMET50_DZ_PFMHT60_MUmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET140_PFMHT140_IDTight_v9")



susHLTSOSmonitoring = cms.Sequence(
 DoubleMu3_PFMET50_DZ_PFMHT60_MUmonitoring+
DoubleMu3_PFMET50_DZ_PFMHT60_METmonitoring
 
)

