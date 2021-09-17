import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ObjMonitor_cfi import hltobjmonitoring

# HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v* and 
VBFSUSYmonitoring = hltobjmonitoring.clone()
VBFSUSYmonitoring.FolderName = cms.string('HLT/SUSY/VBF/DiJet/')
VBFSUSYmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults","","HLT" )
VBFSUSYmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v*","HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v*")
VBFSUSYmonitoring.jetSelection = cms.string("pt>40 & abs(eta)<5.0")
VBFSUSYmonitoring.jetId = cms.string("loose")
VBFSUSYmonitoring.njets = cms.int32(2)
#VBFSUSYmonitoring.enableMETPlot = True
#VBFSUSYmonitoring.metSelection = cms.string("pt>50")
VBFSUSYmonitoring.htjetSelection = cms.string("pt>30 & abs(eta)<5.0")

susyHLTVBFMonitoring = cms.Sequence(
    VBFSUSYmonitoring
)

