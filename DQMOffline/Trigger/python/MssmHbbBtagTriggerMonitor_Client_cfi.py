import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

mssmHbbBtag = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/MssmHbb/semileptonic/BtagTrigger/pt40_noMuon/"),
    subDirs        = cms.untracked.vstring("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_BtagPT          'BTag rel eff vs pT;            probe pT [GeV]; efficiency'     pt_probe_match          pt_probe",
        "effic_BtagETA         'BTag rel eff vs eta;           probe eta; efficiency'          eta_probe_match        eta_probe",
        "effic_BtagPHI         'BTag rel eff vs phi;           probe phi; efficiency'          phi_probe_match        phi_probe",
    ),
)

mssmHbbBtagSL40noMu = mssmHbbBtag.clone()
mssmHbbBtagSL40noMu.subDirs = cms.untracked.vstring("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/pt40_noMuon")

mssmHbbBtagSL40 = mssmHbbBtag.clone()
#mssmHbbBtagSL40.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/semileptonic/BtagTrigger/pt40/")
mssmHbbBtagSL40.subDirs = cms.untracked.vstring("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/pt40/")

mssmHbbBtagSL100 = mssmHbbBtag.clone()
#mssmHbbBtagSL100.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/semileptonic/BtagTrigger/pt100/")
mssmHbbBtagSL100.subDirs = cms.untracked.vstring("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/pt100/")

mssmHbbBtagSL200 = mssmHbbBtag.clone()
#mssmHbbBtagSL200.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/semileptonic/BtagTrigger/pt200/")
mssmHbbBtagSL200.subDirs = cms.untracked.vstring("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/pt200/")

mssmHbbBtagSL350 = mssmHbbBtag.clone()
#mssmHbbBtagSL350.subDirs = cms.untracked.vstring("HLT/Higgs/MssmHbb/semileptonic/BtagTrigger/pt350/")
mssmHbbBtagSL350.subDirs = cms.untracked.vstring("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/pt350/")

mssmHbbBtagAH100 = mssmHbbBtag.clone()
mssmHbbBtagAH100.subDirs = cms.untracked.vstring("HLT/HIG/MssmHbb/fullhadronic/BtagTrigger/pt100/")

mssmHbbBtagAH200 = mssmHbbBtag.clone()
mssmHbbBtagAH200.subDirs = cms.untracked.vstring("HLT/HIG/MssmHbb/fullhadronic/BtagTrigger/pt200/")

mssmHbbBtagAH350 = mssmHbbBtag.clone()
mssmHbbBtagAH350.subDirs = cms.untracked.vstring("HLT/HIG/MssmHbb/fullhadronic/BtagTrigger/pt350/")



mssmHbbBtagTriggerEfficiency = cms.Sequence(
   mssmHbbBtag
#   mssmHbbBtagSL40noMu
# + mssmHbbBtagSL40
# + mssmHbbBtagSL100
# + mssmHbbBtagSL200
# + mssmHbbBtagSL350
# + mssmHbbBtagAH100
# + mssmHbbBtagAH200
# + mssmHbbBtagAH350
)
