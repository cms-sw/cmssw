import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.razorHemispheres_cff import *
from DQMOffline.Trigger.RazorMonitor_cfi import hltRazorMonitoring

# HLT_Rsq0p35_v* 
Rsq0p35_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/Rsq0p35/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Rsq0p35_v*"])
)
# HLT_Rsq0p35_v* tight
Rsq0p35_Tight_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/Rsq0p35_Tight/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Rsq0p35_v*"]),
        jetSelection = "pt>120",
)
# HLT_Rsq0p40_v*
Rsq0p40_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/Rsq0p40/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Rsq0p40_v*"])
)
# HLT_Rsq0p40_v* tight
Rsq0p40_Tight_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/Rsq0p40_Tight/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Rsq0p40_v*"]),
        jetSelection = "pt>120"
)
# HLT_RsqMR300_Rsq0p09_MR200_v*
RsqMR300_Rsq0p09_MR200_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/RsqMR300_Rsq0p09_MR200/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_RsqMR300_Rsq0p09_MR200_v*"])
)
# HLT_RsqMR300_Rsq0p09_MR200_v* tight
RsqMR300_Rsq0p09_MR200_Tight_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/RsqMR300_Rsq0p09_MR200_Tight/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_RsqMR300_Rsq0p09_MR200_v*"]),
        jetSelection = "pt>120"
)
# HLT_RsqMR320_Rsq0p09_MR200_v*
RsqMR320_Rsq0p09_MR200_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/RsqMR320_Rsq0p09_MR200/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_RsqMR320_Rsq0p09_MR200_v*"])
)
# HLT_RsqMR320_Rsq0p09_MR200_v* tight
RsqMR320_Rsq0p09_MR200_Tight_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/RsqMR320_Rsq0p09_MR200_Tight/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_RsqMR320_Rsq0p09_MR200_v*"]),
        jetSelection = "pt>120"
)
# HLT_RsqMR300_Rsq0p09_MR200_4jet_v*
RsqMR300_Rsq0p09_MR200_4jet_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/RsqMR300_Rsq0p09_MR200_4jet/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_RsqMR300_Rsq0p09_MR200_4jet_v*"])
)
# HLT_RsqMR300_Rsq0p09_MR200_4jet_v* tight
RsqMR300_Rsq0p09_MR200_4jet_Tight_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/RsqMR300_Rsq0p09_MR200_4jet_Tight/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_RsqMR300_Rsq0p09_MR200_4jet_v*"]),
        jetSelection = "pt>120"
)
# HLT_RsqMR320_Rsq0p09_MR200_4jet_v*
RsqMR320_Rsq0p09_MR200_4jet_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/RsqMR320_Rsq0p09_MR200_4jet/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_RsqMR320_Rsq0p09_MR200_4jet_v*"])
)
# HLT_RsqMR320_Rsq0p09_MR200_4jet_v* tight
RsqMR320_Rsq0p09_MR200_4jet_Tight_RazorMonitoring = hltRazorMonitoring.clone(
        FolderName = 'HLT/SUSY/RsqMR320_Rsq0p09_MR200_4jet_Tight/',
        numGenericTriggerEventPSet = dict(hltPaths = ["HLT_RsqMR320_Rsq0p09_MR200_4jet_v*"]),
        jetSelection = "pt>120"
)
susyHLTRazorMonitoring = cms.Sequence(
        cms.ignore(hemispheresDQM)+ #for razor triggers
        cms.ignore(caloHemispheresDQM)+ #for razor triggers
        Rsq0p35_RazorMonitoring+
        Rsq0p35_Tight_RazorMonitoring+
        Rsq0p40_RazorMonitoring+
        Rsq0p40_Tight_RazorMonitoring+
        RsqMR300_Rsq0p09_MR200_RazorMonitoring+
        RsqMR300_Rsq0p09_MR200_Tight_RazorMonitoring+
        RsqMR320_Rsq0p09_MR200_RazorMonitoring+
        RsqMR320_Rsq0p09_MR200_Tight_RazorMonitoring+
        RsqMR300_Rsq0p09_MR200_4jet_RazorMonitoring+
        RsqMR300_Rsq0p09_MR200_4jet_Tight_RazorMonitoring+
        RsqMR320_Rsq0p09_MR200_4jet_RazorMonitoring+
        RsqMR320_Rsq0p09_MR200_4jet_Tight_RazorMonitoring
)

