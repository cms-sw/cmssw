import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

mssmHbbBtagSL40noMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt40_noMuon/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_mssmhbbsl40nomuPT          'BTag rel eff vs pT;            probe pT [GeV]; efficiency'     pt_probe_match          pt_probe",
        "effic_mssmhbbsl40nomuETA         'BTag rel eff vs eta;           probe eta; efficiency'          eta_probe_match        eta_probe",
        "effic_mssmhbbsl40nomuPHI         'BTag rel eff vs phi;           probe phi; efficiency'          phi_probe_match        phi_probe",
    ),
)

mssmHbbBtagTriggerEfficiency = cms.Sequence(
   mssmHbbBtagSL40noMu
)
