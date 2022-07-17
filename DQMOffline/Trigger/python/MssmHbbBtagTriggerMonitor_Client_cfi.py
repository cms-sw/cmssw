import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

mssmHbbBtag = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring(
        "HLT/HIG/MssmHbb/control/btag/*",
    ),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_BtagPT          'BTag rel eff vs pT;            probe pT [GeV]; efficiency'     pt_probe_match          pt_probe",
        "effic_BtagETA         'BTag rel eff vs eta;           probe eta; efficiency'          eta_probe_match        eta_probe",
        "effic_BtagPHI         'BTag rel eff vs phi;           probe phi; efficiency'          phi_probe_match        phi_probe",
    ),
)

mssmHbbBtagTriggerEfficiency = cms.Sequence(
   mssmHbbBtag 
)
