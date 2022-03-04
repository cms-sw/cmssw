import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ObjMonitor_cfi import hltobjmonitoring

# HLT_
DiJetVBFmonitoring = hltobjmonitoring.clone(
    FolderName = 'HLT/HIG/VBFMET/DiJet/',
    numGenericTriggerEventPSet = dict(hltInputTag   = "TriggerResults::HLT",
                                      hltPaths = ["HLT_DiJet110_35_Mjj650_PFMET110_v*","HLT_DiJet110_35_Mjj650_PFMET120_v*","HLT_DiJet110_35_Mjj650_PFMET130_v*"]),
    jetSelection = "pt>40 & abs(eta)<4.7",
    jetId = "loose",
    njets = 2
    #enableMETPlot = True
    #metSelection = "pt>150",
)
TripleJetVBFmonitoring = DiJetVBFmonitoring.clone(
    FolderName = 'HLT/HIG/VBFMET/TripleJet/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_TripleJet110_35_35_Mjj650_PFMET110_v*","HLT_TripleJet110_35_35_Mjj650_PFMET120_v*","HLT_TripleJet110_35_35_Mjj650_PFMET130_v*"])
)
higgsinvHLTJetMETmonitoring = cms.Sequence(
    DiJetVBFmonitoring
    *TripleJetVBFmonitoring
)
