import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ObjMonitor_cfi import hltobjmonitoring

# HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v* and 
VBFSUSmonitoring = hltobjmonitoring.clone(
    FolderName = 'HLT/SUS/VBF/DiJet/',
    numGenericTriggerEventPSet = dict(hltInputTag = "TriggerResults::HLT" ,
                                  hltPaths = ["HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v*","HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v*"]),
    jetSelection = "pt>40 & abs(eta)<5.0",
    jetId = "loose",
    njets = 2,
    #enableMETPlot = True,
    #metSelection = "pt>50",
    htjetSelection = "pt>30 & abs(eta)<5.0"
)
susHLTVBFMonitoring = cms.Sequence(
    VBFSUSmonitoring
)

