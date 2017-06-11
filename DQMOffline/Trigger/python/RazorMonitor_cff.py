import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.razorHemispheres_cff import *
from DQMOffline.Trigger.RazorMonitor_cfi import hltRazorMonitoring

# HLT_Rsq0p25_v*
Rsq0p25_RazorMonitoring = hltRazorMonitoring.clone()
Rsq0p25_RazorMonitoring.FolderName = cms.string('HLT/SUSY/Rsq0p25/')
Rsq0p25_RazorMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Rsq0p25_v*")

# HLT_Rsq0p30_v*
Rsq0p30_RazorMonitoring = hltRazorMonitoring.clone()
Rsq0p30_RazorMonitoring.FolderName = cms.string('HLT/SUSY/Rsq0p30/')
Rsq0p30_RazorMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Rsq0p30_v*")

# HLT_RsqMR270_Rsq0p09_MR200_v*
RsqMR270_Rsq0p09_MR200_RazorMonitoring = hltRazorMonitoring.clone()
RsqMR270_Rsq0p09_MR200_RazorMonitoring.FolderName = cms.string('HLT/SUSY/RsqMR270_Rsq0p09_MR200/')
RsqMR270_Rsq0p09_MR200_RazorMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_RsqMR270_Rsq0p09_MR200_v*")

# HLT_RsqMR300_Rsq0p09_MR200_v*
RsqMR300_Rsq0p09_MR200_RazorMonitoring = hltRazorMonitoring.clone()
RsqMR300_Rsq0p09_MR200_RazorMonitoring.FolderName = cms.string('HLT/SUSY/RsqMR300_Rsq0p09_MR200/')
RsqMR300_Rsq0p09_MR200_RazorMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_RsqMR300_Rsq0p09_MR200_v*")

# HLT_RsqMR320_Rsq0p09_MR200_v*
RsqMR320_Rsq0p09_MR200_RazorMonitoring = hltRazorMonitoring.clone()
RsqMR320_Rsq0p09_MR200_RazorMonitoring.FolderName = cms.string('HLT/SUSY/RsqMR320_Rsq0p09_MR200/')
RsqMR320_Rsq0p09_MR200_RazorMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_RsqMR320_Rsq0p09_MR200_v*")

# HLT_RsqMR270_Rsq0p09_MR200_4jet_v*
RsqMR270_Rsq0p09_MR200_4jet_RazorMonitoring = hltRazorMonitoring.clone()
RsqMR270_Rsq0p09_MR200_4jet_RazorMonitoring.FolderName = cms.string('HLT/SUSY/RsqMR270_Rsq0p09_MR200_4jet/')
RsqMR270_Rsq0p09_MR200_4jet_RazorMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_RsqMR270_Rsq0p09_MR200_4jet_v*")

# HLT_RsqMR300_Rsq0p09_MR200_4jet_v*
RsqMR300_Rsq0p09_MR200_4jet_RazorMonitoring = hltRazorMonitoring.clone()
RsqMR300_Rsq0p09_MR200_4jet_RazorMonitoring.FolderName = cms.string('HLT/SUSY/RsqMR300_Rsq0p09_MR200_4jet/')
RsqMR300_Rsq0p09_MR200_4jet_RazorMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_RsqMR300_Rsq0p09_MR200_4jet_v*")

# HLT_RsqMR320_Rsq0p09_MR200_4jet_v*
RsqMR320_Rsq0p09_MR200_4jet_RazorMonitoring = hltRazorMonitoring.clone()
RsqMR320_Rsq0p09_MR200_4jet_RazorMonitoring.FolderName = cms.string('HLT/SUSY/RsqMR320_Rsq0p09_MR200_4jet/')
RsqMR320_Rsq0p09_MR200_4jet_RazorMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_RsqMR320_Rsq0p09_MR200_4jet_v*")


susyHLTRazorMonitoring = cms.Sequence(
        cms.ignore(hemispheresDQM)+ #for razor triggers
        cms.ignore(caloHemispheresDQM)+ #for razor triggers
        Rsq0p25_RazorMonitoring+
        Rsq0p30_RazorMonitoring+
        RsqMR270_Rsq0p09_MR200_RazorMonitoring+
        RsqMR300_Rsq0p09_MR200_RazorMonitoring+
        RsqMR320_Rsq0p09_MR200_RazorMonitoring+
        RsqMR270_Rsq0p09_MR200_4jet_RazorMonitoring+
        RsqMR300_Rsq0p09_MR200_4jet_RazorMonitoring+
        RsqMR320_Rsq0p09_MR200_4jet_RazorMonitoring
)

