import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

mssmHbbBtagSL40noMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Higgs/MssmHbb/semileptonic/BtagTrigger/pt40_noMuon/"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_BtagPT          'BTag rel eff vs pT;            probe pT [GeV]; efficiency'     pt_probe_match          pt_probe",
        "effic_BtagETA         'BTag rel eff vs eta;           probe eta; efficiency'          eta_probe_match        eta_probe",
        "effic_BtagPHI         'BTag rel eff vs phi;           probe phi; efficiency'          phi_probe_match        phi_probe",
    ),
)

mssmHbbBtagSL40 = mssmHbbBtagSL40noMu.clone()
mssmHbbBtagSL40.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/semileptonic/BtagTrigger/pt40/")

mssmHbbBtagSL100 = mssmHbbBtagSL40noMu.clone()
mssmHbbBtagSL100.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/semileptonic/BtagTrigger/pt100/")

mssmHbbBtagSL200 = mssmHbbBtagSL40noMu.clone()
mssmHbbBtagSL200.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/semileptonic/BtagTrigger/pt200/")

mssmHbbBtagSL350 = mssmHbbBtagSL40noMu.clone()
mssmHbbBtagSL350.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/semileptonic/BtagTrigger/pt350/")

mssmHbbBtagAH100 = mssmHbbBtagSL40noMu.clone()
mssmHbbBtagAH100.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/fullhadronic/BtagTrigger/pt100/")

mssmHbbBtagAH200 = mssmHbbBtagSL40noMu.clone()
mssmHbbBtagAH200.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/fullhadronic/BtagTrigger/pt200/")

mssmHbbBtagAH350 = mssmHbbBtagSL40noMu.clone()
mssmHbbBtagAH350.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/fullhadronic/BtagTrigger/pt350/")



mssmHbbBtagTriggerEfficiency = cms.Sequence(
   mssmHbbBtagSL40noMu
 + mssmHbbBtagSL40
 + mssmHbbBtagSL100
 + mssmHbbBtagSL200
 + mssmHbbBtagSL350
 + mssmHbbBtagAH100
 + mssmHbbBtagAH200
 + mssmHbbBtagAH350
)
